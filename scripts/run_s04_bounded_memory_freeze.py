#!/usr/bin/env python3
"""S04 privileged local test of sustained target-memory freeze versus one skipped write."""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import platform
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
import yaml

from analyze_p1_common_state_mechanism import read_detection_cache
from analyze_s02_natural_identity_mechanism import analyze_condition, feature_map
from mot_research.analysis.failure_events import (
    hungarian_iou_matches, read_detections, read_gt, read_tracker,
)
from run_s01_visual_baselines import all_detection_ids, load_features
from run_s03_memory_clone_smoke import MRunner, require_hash, sha256
from run_s03_single_memory_write_counterfactual_retry2 import (
    MultiCVTap, key, score_case, select_cases,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'manifests/protocols/s04_bounded_memory_freeze_20261122.yaml'


def state_equal(a, b) -> bool:
    """Exact recursive comparison of tracker and CV-history state, except callback identity."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, '__dict__'):
        ad = {k: v for k, v in vars(a).items() if k != '_s01_record'}
        bd = {k: v for k, v in vars(b).items() if k != '_s01_record'}
        return state_equal(ad, bd)
    if callable(a):
        return a is b
    if isinstance(a, (str, bytes, int, float, bool, type(None), np.generic)):
        return bool(a == b)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        raise TypeError(f'unsupported state comparison type: {type(a)!r}')


def assert_event_isolation(factual: MRunner, single: MRunner,
                           frozen: MRunner, frame: int, target: int) -> None:
    for branch in (single, frozen):
        if factual.next_count != branch.next_count or factual.mature_ids != branch.mature_ids:
            raise RuntimeError('event-frame ID/mature state changed')
        if not state_equal(factual.tracker, branch.tracker) or \
                not state_equal(factual.history, branch.history):
            raise RuntimeError('event-frame tracker/CV history state changed')
        if factual.memory._last_frame != branch.memory._last_frame or \
                factual.memory._pending != branch.memory._pending:
            raise RuntimeError('event-frame memory lifecycle differs')
        if factual.memory._memory.keys() != branch.memory._memory.keys():
            raise RuntimeError('event-frame memory key set differs')
        for tid, (time, value) in factual.memory._memory.items():
            other_time, other_value = branch.memory._memory[tid]
            if tid != target and (time != other_time or not np.array_equal(value, other_value)):
                raise RuntimeError('non-target memory changed')
        if factual.memory._memory[target][0] != frame or \
                branch.memory._memory[target][0] >= frame or \
                np.array_equal(factual.memory._memory[target][1],
                               branch.memory._memory[target][1]):
            raise RuntimeError('selected target memory was not isolated')
    if not state_equal(single.memory._memory, frozen.memory._memory):
        raise RuntimeError('single/frozen event memory differs')


class TargetEdgeTap:
    """Observe actual CV-adapted target-column legal edges without altering assignment."""

    def __init__(self, base, visual, assoc_module, target: int):
        self.base, self.visual, self.assoc_module, self.target = base, visual, assoc_module, target
        self.by_frame: dict[int, int] = {}

    def __call__(self, request, associate):
        def observed(dets, trks, threshold, velocities, observations, inertia):
            count = 0
            if self.target in request.tracker_ids and len(dets) and len(trks):
                iou = self.assoc_module.iou_batch(dets, trks)
                count = int(np.count_nonzero(iou[:, request.tracker_ids.index(self.target)] >= .3))
            self.by_frame[request.frame_id] = count
            return associate(dets, trks, threshold, velocities, observations, inertia)
        return self.base(request, observed)


def attach_target_tap(runner: MRunner, target: int) -> TargetEdgeTap:
    tap = TargetEdgeTap(runner.visual.cv_policy, runner.visual, runner.assoc_module, target)
    runner.visual.cv_policy = tap
    return tap


def freeze_future_writes(runner: MRunner, target: int,
                         removed: list[tuple[int, int]]) -> None:
    original = runner.memory.finish_frame

    def filtered(frame, live_track_ids, matched_track_to_det, det_ids, detection_features):
        mapping = dict(matched_track_to_det)
        if target in mapping:
            removed.append((int(frame), int(mapping.pop(target))))
        return original(frame, live_track_ids, mapping, det_ids, detection_features)

    runner.memory.finish_frame = filtered


def run_branch(case: dict, snapshot: MRunner, last_frame: int,
               source_rows: dict[int, str], source_traces: list[str]) -> dict:
    first, target = case['frame'], case['track_id']
    end = min(last_frame, first + 30)
    factual, single, frozen = snapshot.fork(), snapshot.fork(), snapshot.fork()
    taps = {name: attach_target_tap(runner, target) for name, runner in
            [('factual', factual), ('single', single), ('frozen', frozen)]}
    rows = {name: [] for name in taps}
    traces = {name: [] for name in taps}
    removed = [(first, case['selected_det_id'])]
    for frame in range(first, end + 1):
        for name, runner in [('factual', factual), ('single', single), ('frozen', frozen)]:
            suppress = target if frame == first and name != 'factual' else None
            value, trace = runner.step(frame, suppress)
            rows[name].append(value)
            traces[name].append(trace)
            if name == 'factual' and (value != source_rows.get(frame, '') or
                                     trace != source_traces[frame - 1]):
                raise RuntimeError(f'factual branch differs from frozen M: {key(case)}/{frame}')
        if frame == first:
            if rows['factual'][-1] != rows['single'][-1] or \
                    rows['factual'][-1] != rows['frozen'][-1] or \
                    traces['factual'][-1] != traces['single'][-1] or \
                    traces['factual'][-1] != traces['frozen'][-1]:
                raise RuntimeError(f'event-frame output/association changed: {key(case)}')
            assert_event_isolation(factual, single, frozen, frame, target)
            freeze_future_writes(frozen, target, removed)
    exposures = {}
    for name, tap in taps.items():
        exposures[name] = {
            'target_legal_edge_frames': sum(tap.by_frame.get(frame, 0) > 0
                                            for frame in range(first + 1, end + 1)),
            'target_legal_edges': sum(tap.by_frame.get(frame, 0)
                                      for frame in range(first + 1, end + 1)),
            'target_nonfast_legal_edge_frames': sum(
                tap.by_frame.get(frame, 0) > 0 and
                not json.loads(traces[name][frame - first])['fast_path']
                for frame in range(first + 1, end + 1)),
        }
    return {'case': case, 'last_frame': end,
            'factual_rows': rows['factual'], 'single_rows': rows['single'],
            'frozen_rows': rows['frozen'],
            'factual_traces': traces['factual'], 'single_traces': traces['single'],
            'frozen_traces': traces['frozen'],
            'frozen_suppressed_writes': [[frame, did] for frame, did in removed],
            'future_frames_with_single_output_change': sum(
                a != b for a, b in zip(rows['factual'][1:], rows['single'][1:])),
            'future_frames_with_frozen_output_change': sum(
                a != b for a, b in zip(rows['factual'][1:], rows['frozen'][1:])),
            'future_frames_with_frozen_minus_single_output_change': sum(
                a != b for a, b in zip(rows['single'][1:], rows['frozen'][1:])),
            'future_target_edge_exposure': exposures}


def score_three_arms(branch: dict, gt: dict,
                     prior_outputs: dict) -> dict:
    case = branch['case']
    common = {'case': case, 'factual_rows': branch['factual_rows']}
    single = score_case({**common, 'suppressed_rows': branch['single_rows'],
                         'future_frames_with_changed_rows': branch['future_frames_with_single_output_change']},
                        gt, prior_outputs)
    frozen = score_case({**common, 'suppressed_rows': branch['frozen_rows'],
                         'future_frames_with_changed_rows': branch['future_frames_with_frozen_output_change']},
                        gt, prior_outputs)
    if (single['prior_output_anchor_count'] != frozen['prior_output_anchor_count'] or
            single['owner_in_prior_output_anchors'] != frozen['owner_in_prior_output_anchors']):
        raise RuntimeError('the two comparisons have different original anchors')
    per_anchor = {gid: frozen['per_anchor_delta'][gid] - value
                  for gid, value in single['per_anchor_delta'].items()}
    return {'case_id': key(case), 'kind': case['kind'], 'sequence': case['sequence'],
            'frame': case['frame'], 'track_id': case['track_id'],
            'owner_gt_id': case['owner_gt_id'],
            'matched_control_id': case.get('matched_control_id'),
            'matched_wrong_id': case.get('matched_wrong_id'),
            'match_cost': case.get('match_cost'),
            'prior_output_anchor_count': single['prior_output_anchor_count'],
            'owner_in_prior_output_anchors': single['owner_in_prior_output_anchors'],
            'future_frames_scored': single['future_frames_scored'],
            'single_minus_factual': {k: single[k] for k in
                ('target_correct_frame_delta', 'all_prior_anchor_correct_frame_delta',
                 'other_prior_anchor_correct_frame_delta', 'other_prior_anchor_harmed',
                 'per_anchor_delta')},
            'frozen_minus_factual': {k: frozen[k] for k in
                ('target_correct_frame_delta', 'all_prior_anchor_correct_frame_delta',
                 'other_prior_anchor_correct_frame_delta', 'other_prior_anchor_harmed',
                 'per_anchor_delta')},
            'frozen_minus_single': {
                'target_correct_frame_delta': frozen['target_correct_frame_delta'] -
                                              single['target_correct_frame_delta'],
                'all_prior_anchor_correct_frame_delta': sum(per_anchor.values()),
                'other_prior_anchor_correct_frame_delta': sum(
                    value for gid, value in per_anchor.items()
                    if int(gid) != case['owner_gt_id']),
                'other_prior_anchor_harmed': any(
                    value < 0 for gid, value in per_anchor.items()
                    if int(gid) != case['owner_gt_id']),
                'per_anchor_delta': per_anchor},
            'frozen_suppressed_write_count': len(branch['frozen_suppressed_writes']),
            'future_frames_with_single_output_change': branch['future_frames_with_single_output_change'],
            'future_frames_with_frozen_output_change': branch['future_frames_with_frozen_output_change'],
            'future_frames_with_frozen_minus_single_output_change':
                branch['future_frames_with_frozen_minus_single_output_change'],
            'future_target_edge_exposure': branch['future_target_edge_exposure']}


def summarize(scores: list[dict], sequences: list[str]) -> dict:
    groups = {}
    per_video = []
    for seq in sequences:
        video = {'sequence': seq, 'groups': {}}
        for kind in ('wrong', 'clean'):
            items = [s for s in scores if s['sequence'] == seq and s['kind'] == kind]
            video['groups'][kind] = {
                'cases': len(items),
                'informative_cases': sum(s['frozen_suppressed_write_count'] >= 2 for s in items),
                'single_changed_cases': sum(s['future_frames_with_single_output_change'] > 0 for s in items),
                'frozen_changed_cases': sum(s['future_frames_with_frozen_output_change'] > 0 for s in items),
                'frozen_minus_single_changed_cases': sum(
                    s['future_frames_with_frozen_minus_single_output_change'] > 0 for s in items),
                'single_minus_factual_all_anchor_delta': sum(
                    s['single_minus_factual']['all_prior_anchor_correct_frame_delta'] for s in items),
                'frozen_minus_factual_all_anchor_delta': sum(
                    s['frozen_minus_factual']['all_prior_anchor_correct_frame_delta'] for s in items),
                'frozen_minus_single_all_anchor_delta': sum(
                    s['frozen_minus_single']['all_prior_anchor_correct_frame_delta'] for s in items),
                'frozen_minus_single_informative_all_anchor_delta': sum(
                    s['frozen_minus_single']['all_prior_anchor_correct_frame_delta']
                    for s in items if s['frozen_suppressed_write_count'] >= 2),
                'frozen_minus_single_other_harmed_cases': sum(
                    s['frozen_minus_single']['other_prior_anchor_harmed'] for s in items),
                'target_legal_exposure_cases': sum(
                    s['future_target_edge_exposure']['frozen']['target_nonfast_legal_edge_frames'] > 0
                    for s in items),
            }
        per_video.append(video)
    for kind in ('wrong', 'clean'):
        groups[kind] = {field: sum(row['groups'][kind][field] for row in per_video)
                        for field in per_video[0]['groups'][kind]}
    wrong, clean = groups['wrong'], groups['clean']
    pair_fraction = clean['cases'] / wrong['cases'] if wrong['cases'] else 0.0
    sparse = (wrong['cases'] < 10 or
              sum(row['groups']['wrong']['cases'] > 0 for row in per_video) < 4 or
              wrong['informative_cases'] < 8 or pair_fraction < .8)
    informative_rate = (wrong['frozen_minus_single_informative_all_anchor_delta'] /
                        wrong['informative_cases']) if wrong['informative_cases'] else None
    clean_rate = (clean['frozen_minus_factual_all_anchor_delta'] /
                  clean['cases']) if clean['cases'] else None
    positive_videos = sum(row['groups']['wrong']['frozen_minus_single_all_anchor_delta'] > 0
                          for row in per_video)
    signal = (not sparse and informative_rate >= .25 and positive_videos >= 4 and
              wrong['frozen_minus_single_other_harmed_cases'] <=
              clean['frozen_minus_single_other_harmed_cases'] and clean_rate >= -.25)
    return {'per_video': per_video, 'totals': groups,
            'wrong_event_videos': sum(row['groups']['wrong']['cases'] > 0 for row in per_video),
            'positive_wrong_freeze_minus_single_videos': positive_videos,
            'control_pair_fraction': pair_fraction,
            'informative_wrong_delta_per_case': informative_rate,
            'clean_freeze_delta_per_case': clean_rate,
            'decision': 'SPARSE_NO_TRAINING' if sparse else
                        ('MECHANISM_SIGNAL_REVIEW_BEFORE_TRAINING' if signal else
                         'NO_SIGNAL_STOP_CURRENT_RATIONALE')}


def overlap_summary(cases: list[dict], lengths: dict[str, int]) -> dict:
    exposure = Counter()
    truncated = []
    for case in cases:
        last = min(lengths[case['sequence']], case['frame'] + 30)
        if last - case['frame'] < 30:
            truncated.append(key(case))
        for frame in range(case['frame'] + 1, last + 1):
            exposure[(case['sequence'], frame)] += 1
    return {'future_frame_exposures': sum(exposure.values()),
            'unique_video_frames': len(exposure),
            'reused_video_frames': sum(n > 1 for n in exposure.values()),
            'maximum_multiplicity': max(exposure.values(), default=0),
            'truncated_case_ids': truncated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-lock', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('S04 branch runner requires CUDA_VISIBLE_DEVICES empty')
    lock = json.loads(args.runtime_lock.read_text())
    if lock['protocol_sha256'] != sha256(PROTOCOL) or \
            lock['analysis_code_sha256'] != sha256(Path(__file__)):
        raise RuntimeError('S04 protocol/code changed after lock')
    for rel, expected in lock['dependency_code_sha256'].items():
        require_hash(ROOT / rel, expected)
    actual_versions = {'python': platform.python_version(), 'numpy': np.__version__,
                       'scipy': scipy.__version__, 'pyyaml': yaml.__version__}
    if actual_versions != lock['environment_versions']:
        raise RuntimeError(f'S04 environment changed: {actual_versions}')
    plan = yaml.safe_load(PROTOCOL.read_text())
    if lock['sequences'] != plan['source_and_scope']['sequences'] or \
            lock['output_root'] != plan['baseline_freeze']['output_root'] + '/branches':
        raise RuntimeError('S04 scope/output differs from lock')
    require_hash(ROOT / lock['baseline_runtime_lock_path'], lock['baseline_runtime_lock_sha256'])
    require_hash(ROOT / lock['feature_runtime_lock_path'], lock['feature_runtime_lock_sha256'])
    baseline_master = json.loads((ROOT / lock['baseline_runtime_lock_path']).read_text())
    feature_lock = json.loads((ROOT / lock['feature_runtime_lock_path']).read_text())
    require_hash(Path(baseline_master['output_root']) / 'status.jsonl',
                 lock['baseline_status_sha256'])
    require_hash(Path(feature_lock['output_root']) / 'status.jsonl',
                 lock['feature_status_sha256'])
    require_hash(Path(feature_lock['input_image_manifest_path']),
                 feature_lock['input_image_manifest_sha256'])
    if baseline_master['selected_sequences'] != lock['sequences'] or \
            feature_lock['selected_sequences'] != lock['sequences']:
        raise RuntimeError('upstream sequence order differs')
    output = Path(lock['output_root'])
    if output.exists():
        raise FileExistsError(output)
    source = {}
    source_record = {'status': 'S04_baseline_sources_frozen_before_GT_read',
                     'protocol_sha256': sha256(PROTOCOL),
                     'runtime_lock_sha256': sha256(args.runtime_lock),
                     'sequences': lock['sequences'], 'artifacts': []}
    for seq in lock['sequences']:
        folder = Path(baseline_master['output_root']) / seq
        paths = {'tracker': folder / 'trackers/M' / f'{seq}.txt',
                 'trace': folder / 'traces/M' / f'{seq}.jsonl',
                 'feature': Path(feature_lock['output_root']) / f'{seq}.npz',
                 'detection': Path(feature_lock['detection_root']) / f'{seq}.csv',
                 'baseline_check': folder / 'engineering_checks.json',
                 'baseline_lock': folder / 'runtime_lock.json'}
        for name, path in paths.items():
            expected = lock['source_sha256'][seq][name]
            require_hash(path, expected)
            source_record['artifacts'].append({'sequence': seq, 'kind': name,
                                               'path': str(path), 'sha256': expected})
        source[seq] = paths
    output.mkdir(parents=True)
    (output / 'frozen_sources.json').write_text(json.dumps(source_record, indent=2) + '\n')

    # All GT-derived selection is offline. Branch inference receives only the
    # target track ID and never a future GT label or GT-positive detection.
    contexts = {}
    selections = {}
    census = {}
    for seq in lock['sequences']:
        gt_path = Path(plan['source_and_scope']['train_root']) / seq / 'gt/gt.txt'
        require_hash(gt_path, feature_lock['input_gt_sha256'][seq])
        gt = read_gt(gt_path)
        det = read_detections(source[seq]['detection'], .6)
        traces = [json.loads(line) for line in source[seq]['trace'].read_text().splitlines()]
        features = feature_map(source[seq]['feature'])
        positives, labels = {}, {}
        for frame in range(1, len(traces) + 1):
            matches, _ = hungarian_iou_matches(gt.get(frame, {}), det.get(frame, []), .5)
            positives[frame] = {gid: did for gid, (did, _) in matches.items()}
            labels[frame] = {did: gid for gid, did in positives[frame].items()}
        counts, events = analyze_condition(seq, 'M', traces, positives, labels, features)
        wrong = [event for event in events if event['actionable_competition']]
        cached, _ = read_detection_cache(source[seq]['detection'])
        dids = all_detection_ids(source[seq]['detection'])
        source_rows = {frame: {did: np.asarray(row, np.float64)
                               for did, row in zip(dids.get(frame, []), cached.get(frame, []))}
                       for frame in range(1, len(traces) + 1)}
        selections[seq] = select_cases(seq, wrong,
                                       [int(event['frame']) for event in events],
                                       traces, gt, det, source_rows)
        contexts[seq] = {'gt': gt, 'cached': cached, 'dids': dids, 'traces': traces}
        census[seq] = counts
    cases = [case for seq in lock['sequences'] for case in selections[seq]]
    (output / 'frozen_event_selection.json').write_text(json.dumps({
        'status': 'selected_before_any_branch_output',
        's02_style_census': census, 'cases': cases,
        'wrong_count': sum(case['kind'] == 'wrong' for case in cases),
        'clean_count': sum(case['kind'] == 'clean' for case in cases)}, indent=2) + '\n')

    sys.path.insert(0, str((ROOT / 'third_party/OC_SORT').resolve()))
    import trackers.ocsort_tracker.ocsort as ocsort_module
    import trackers.ocsort_tracker.association as assoc_module
    branch_artifacts = []
    engineering = []
    for seq in lock['sequences']:
        folder = output / seq
        folder.mkdir()
        try:
            cfg = configparser.ConfigParser()
            if not cfg.read(Path(plan['source_and_scope']['train_root']) / seq / 'seqinfo.ini'):
                raise FileNotFoundError(seq)
            info = cfg['Sequence']
            height, width, fps, length = (info.getint('imHeight'), info.getint('imWidth'),
                                          info.getfloat('frameRate'), info.getint('seqLength'))
            context = contexts[seq]
            if len(context['traces']) != length:
                raise RuntimeError('baseline trace/seqinfo length mismatch')
            source_rows: dict[int, str] = defaultdict(str)
            for line in source[seq]['tracker'].read_text().splitlines(keepends=True):
                source_rows[int(line.split(',', 1)[0])] += line
            trace_lines = source[seq]['trace'].read_text().splitlines(keepends=True)
            selected = selections[seq]
            frames = {case['frame'] for case in selected}
            runner = MRunner(seq, context['cached'], context['dids'],
                             load_features(source[seq]['feature']), height, width,
                             fps, ocsort_module, assoc_module, None)
            tap = MultiCVTap(runner.visual.cv_policy, runner.visual, assoc_module, selected)
            runner.visual.cv_policy = tap
            snapshots = {}
            with (folder / 'full_replay_tracker.txt').open('w') as th, \
                    (folder / 'full_replay_trace.jsonl').open('w') as trh:
                for frame in range(1, length + 1):
                    if frame in frames:
                        snapshots[frame] = runner.fork()
                    row, trace = runner.step(frame)
                    if row != source_rows.get(frame, '') or trace != trace_lines[frame - 1]:
                        raise RuntimeError(f'MRunner/S01 M mismatch at {seq}:{frame}')
                    th.write(row)
                    trh.write(trace)
            require_hash(folder / 'full_replay_tracker.txt', sha256(source[seq]['tracker']))
            require_hash(folder / 'full_replay_trace.jsonl', sha256(source[seq]['trace']))
            branch_path = folder / 'branches.jsonl'
            with branch_path.open('w') as handle:
                for case in selected:
                    result = run_branch(case, snapshots[case['frame']], length,
                                        source_rows, trace_lines)
                    result['first_round_legal_screen'] = tap.capture.get(key(case))
                    if result['first_round_legal_screen'] is None:
                        raise RuntimeError('case first-round request not captured')
                    handle.write(json.dumps(result, separators=(',', ':')) + '\n')
            record = {'sequence': seq, 'status': 'ENGINEERING_PASS',
                      'frames_replayed': length, 'cases': len(selected),
                      'tracker_sha256': sha256(folder / 'full_replay_tracker.txt'),
                      'trace_sha256': sha256(folder / 'full_replay_trace.jsonl'),
                      'branches_sha256': sha256(branch_path)}
            engineering.append(record)
            (folder / 'engineering_checks.json').write_text(json.dumps(record, indent=2) + '\n')
            branch_artifacts.append({'sequence': seq, 'path': str(branch_path),
                                     'sha256': record['branches_sha256'],
                                     'case_count': len(selected)})
        except Exception as exc:
            failure = {'sequence': seq, 'status': 'ENGINEERING_FAIL',
                       'reason': repr(exc), 'traceback': traceback.format_exc()}
            (folder / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
            (output / 'partial_engineering.json').write_text(json.dumps(
                engineering + [failure], indent=2) + '\n')
            raise
    (output / 'branches_frozen.json').write_text(json.dumps({
        'status': 'all_branch_outputs_frozen_before_GT_scoring',
        'sources_sha256': sha256(output / 'frozen_sources.json'),
        'selection_sha256': sha256(output / 'frozen_event_selection.json'),
        'artifacts': branch_artifacts}, indent=2) + '\n')

    scores = []
    for seq in lock['sequences']:
        gt = contexts[seq]['gt']
        prior = read_tracker(source[seq]['tracker'])
        for line in (output / seq / 'branches.jsonl').read_text().splitlines():
            branch = json.loads(line)
            scored = score_three_arms(branch, gt, prior)
            scored['first_round_legal_screen'] = branch['first_round_legal_screen']
            scores.append(scored)
    (output / 'scores.jsonl').write_text(''.join(
        json.dumps(item, separators=(',', ':')) + '\n' for item in scores))
    aggregate = summarize(scores, lock['sequences'])
    (output / 'per_video.json').write_text(json.dumps(aggregate['per_video'], indent=2) + '\n')
    summary = {'status': 'COMPLETE_PRIVILEGED_DIAGNOSTIC',
               'scope': '8_train_sequences_disjoint_from_S01_S03_fit_but_used_in_prior_R1',
               'protocol_sha256': sha256(PROTOCOL),
               'runtime_lock_sha256': sha256(args.runtime_lock),
               'frozen_sources_sha256': sha256(output / 'frozen_sources.json'),
               'frozen_event_selection_sha256': sha256(output / 'frozen_event_selection.json'),
               'branches_frozen_sha256': sha256(output / 'branches_frozen.json'),
               'scores_sha256': sha256(output / 'scores.jsonl'),
               'per_video_sha256': sha256(output / 'per_video.json'),
               'overlap': overlap_summary(cases, feature_lock['selected_sequence_lengths']),
               **{k: v for k, v in aggregate.items() if k != 'per_video'},
               'limitations': ['GT-privileged event selection; not a deployable online method',
                               'reused DanceTrack train subset used in prior R1 work',
                               'IoU identity proxy and overlapping local branches',
                               'possible appearance pretraining overlap',
                               'no official full-sequence TrackEval gain from local windows']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'totals': summary['totals'],
                      'decision': summary['decision'],
                      'summary_sha256': sha256(output / 'summary.json')}, indent=2))


if __name__ == '__main__':
    main()
