"""Check Git exports only; no checkpoints, tracking, GPU, or new data."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'reports/diagnostics/s07_s06_checkpoint_postmortem_20261124'
sources = {}


def text(path):
    data = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load(path):
    return json.loads(text(path))


def near(a, b):
    assert abs(a-b) < 1e-12, (a, b)


def weighted_mean(rows, fn):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sequence']].append(fn(row))
    return statistics.mean(statistics.mean(v) for v in groups.values())


artifacts = load(P / 'artifacts.json')
verified = []
for artifact in artifacts['files']:
    if artifact['location'] != 'git':
        continue
    path = P / artifact['path']
    text(path)
    assert sources[str(path.relative_to(ROOT))] == artifact['sha256']
    assert path.stat().st_size == artifact['bytes']
    verified.append(artifact['path'])
manifest = load(P / 'run_manifest.json')
for name, digest in manifest['source_sha256'].items():
    text(ROOT / name)
    assert sources[name] == digest
text(ROOT / manifest['protocol'])
assert sources[manifest['protocol']] == manifest['protocol_sha256']
diagnostics = load(P / 'diagnostics.json')
rows = list(csv.DictReader(text(P / 'checkpoint_metrics.csv').splitlines()))
index = {(r['model'], int(r['epoch']), r['split']): r for r in rows}
assert len(rows) == len(index) == 24
cells = 0
for row in diagnostics['checkpoint_metrics']:
    exported = index[row['model'], row['epoch'], row['split']]
    for key, value in row.items():
        if isinstance(value, (int, float)):
            near(float(exported[key]), value)
            cells += 1
weighting = load(P / 'decision_weighting_addendum.json')
assert len(weighting['results']) == 48
for row in weighting['results']:
    near(statistics.mean(row['per_sequence_accuracy'].values()),
         row['video_balanced_accuracy_protocol_intended'])
    near(statistics.mean(row['always_M_per_sequence_accuracy'].values()),
         row['always_M_video_balanced_accuracy'])

# Rebuild the stronger constants from Git's original S05 labels, without predictions.
protocol = text(ROOT / 'manifests/protocols/s06_component_selector_seed13_20261123.yaml')
train = set(re.search(r'train18:\n    sequences: \[(.*?)\]', protocol).group(1).split(', '))
dev = set(re.search(r'internal_dev6:\n    sequences: \[(.*?)\]', protocol).group(1).split(', '))
assert len(train) == 18 and len(dev) == 6 and not train & dev
labels = [json.loads(line) for line in text(
    ROOT / 'reports/diagnostics/s05_visual_decision_coverage_20261123/component_results.jsonl').splitlines()]
examples = []
for component in labels:
    if not component['fully_labelable']:
        continue
    assert len(component['unique_actions']) == 2
    for action in component['unique_actions']:
        label = component['actions'][action]
        examples.append({'sequence': component['sequence'], 'action': action,
                         'g': label['g']/len(component['tracker_ids']),
                         'h': label['h']/len(component['tracker_ids'])})
fit = [r for r in examples if r['sequence'] in train]
non_m = [r for r in fit if r['action'] != 'M']
means = {a: [weighted_mean([r for r in fit if r['action'] == a], lambda r: r[k])
             for k in ('g', 'h')] for a in ('M', 'G', 'V')}
shared = [weighted_mean(non_m, lambda r: r[k]) for k in ('g', 'h')]
addendum = load(P / 'action_type_controls_addendum.json')
for action in means:
    for actual, reported in zip(means[action], addendum['train_video_balanced_action_type_means_g_h'][action]):
        near(actual, reported)
constant_errors = {}
for split, sequences in [('train18', train), ('internal_dev6', dev)]:
    samples = [r for r in examples if r['sequence'] in sequences]
    constant_errors[split] = {}
    for name, predictor in [
        ('analytic_M_zero_shared_non_M_train_mean', lambda r: [0, 0] if r['action'] == 'M' else shared),
        ('train_action_type_means_M_G_V', lambda r: means[r['action']]),
    ]:
        value = weighted_mean(samples, lambda r: sum((x-r[k])**2 for x,k in zip(predictor(r), ('g','h')))/2)
        near(value, addendum['controls'][split][name]['video_balanced_mean_mse'])
        constant_errors[split][name] = value

context = index['Context', 30, 'internal_dev6']
n, actual, predicted = (int(context[k]) for k in ('components','no_extra_true_non_M','no_extra_predicted_non_M'))
correct = round(n*float(context['no_extra_decision_accuracy']))
# Valid here because every labeled component has exactly M and one alternative.
tp = (correct-n+actual+predicted)//2
confusion = {'TP':tp, 'FP':predicted-tp, 'FN':actual-tp, 'TN':n-actual-predicted+tp}
cases = [json.loads(line) for line in text(P / 'cases.jsonl').splitlines()]
runtime = load(P / 'runtime.json')
csv_bytes = (P / 'checkpoint_metrics.csv').read_bytes()
crlf = hashlib.sha256(csv_bytes.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')).hexdigest()
assert crlf == runtime['output_sha256']['checkpoint_metrics.csv']
text(Path(__file__).resolve())
result = {
    'date':'2026-11-25', 'scope':'local_Git_exports_only', 'new_tracking_or_model_inference':False,
    'verified_git_artifacts':verified, 'verified_source_files':list(manifest['source_sha256']),
    'protocol_hash_matches':True, 'csv_json_numeric_cells':cells,
    'weighting_rows_checked':48, 'original_csv_CRLF_hash_reproduced':True,
    'strong_constants_from_S05_labels':constant_errors,
    'Context_e30_dev_no_extra_binary_decision_confusion':confusion,
    'case_counts_by_sequence':dict(Counter(r['sequence'] for r in cases)),
    'cases_missing_same_state_M':sum(not r['frame_pairing']['same_state_M_candidate_pairs_available'] for r in cases),
    'cases_missing_exact_legal_mask':sum(not r['legality_evidence']['exact_legal_edge_mask_available'] for r in cases),
    'input_sha256':sources,
    'limits':['Raw 3024 predictions and checkpoints are server-only',
              'No new generalization, significance, utility-regret, or causal claim'],
}
destination = Path(__file__).with_name('verification.json')
with destination.open('x') as handle:
    json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    handle.write('\n')
print(json.dumps({'csv_json_cells':cells,'git_hashes':len(verified),'confusion':confusion,'output':str(destination)}))
