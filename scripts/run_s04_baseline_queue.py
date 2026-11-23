#!/usr/bin/env python3
"""Freeze and replay the S04 M baseline separately for each selected video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/home/imc/anaconda3/envs/mot-det/bin/python')
RUNNER = ROOT / 'scripts/run_s01_visual_baselines.py'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-lock', type=Path, required=True)
    args = parser.parse_args()
    master = json.loads(args.runtime_lock.read_text())
    if master['queue_code_sha256'] != sha256(Path(__file__)) or \
            master['protocol_sha256'] != sha256(ROOT / master['protocol_path']):
        raise RuntimeError('S04 queue/protocol differs from lock')
    for rel, expected in master['code_hashes'].items():
        if sha256(ROOT / rel) != expected:
            raise RuntimeError(f'S04 baseline source changed: {rel}')
    if master['feature_lock_sha256'] != sha256(ROOT / master['feature_lock_path']):
        raise RuntimeError('feature extraction lock changed')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('S04 baseline queue requires CUDA_VISIBLE_DEVICES empty')
    output = Path(master['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    status = output / 'status.jsonl'
    with status.open('w') as status_handle:
        failed = 0
        for seq in master['selected_sequences']:
            feature = Path(master['feature_root']) / f'{seq}.npz'
            if sha256(feature) != master['feature_cache_sha256'][seq]:
                raise RuntimeError(f'feature hash mismatch: {seq}')
            if sha256(Path(master['detection_root']) / f'{seq}.csv') != master['input_detection_sha256'][seq]:
                raise RuntimeError(f'detection hash mismatch: {seq}')
            seq_output = output / seq
            seq_output.mkdir()
            lock = {'run_id': f's04_M_baseline_{seq}', 'protocol_sha256': master['protocol_sha256'],
                    'selected_sequences': [seq], 'train_root': master['train_root'],
                    'detection_root': master['detection_root'],
                    'input_detection_sha256': {seq: master['input_detection_sha256'][seq]},
                    'feature_cache_path': str(feature),
                    'feature_cache_sha256': master['feature_cache_sha256'][seq],
                    'code_hashes': master['code_hashes'],
                    'output_root': str(seq_output), 'cuda_visible_devices': '',
                    'physical_gpus': [], 'gpu0_forbidden': True}
            seq_lock = seq_output / 'runtime_lock.json'
            seq_lock.write_text(json.dumps(lock, indent=2) + '\n')
            log = seq_output / 'run.log'
            command = [str(PYTHON), str(RUNNER), '--sequence', seq,
                       '--last-frame', str(master['selected_sequence_lengths'][seq]),
                       '--conditions', 'M', '--runtime-lock', str(seq_lock)]
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1',
                        'PYTHONPATH': 'src:scripts'})
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=handle, stderr=subprocess.STDOUT,
                                        check=False)
            tracker = seq_output / 'trackers/M' / f'{seq}.txt'
            trace = seq_output / 'traces/M' / f'{seq}.jsonl'
            row = {'sequence': seq, 'status': 'PASS' if result.returncode == 0 else 'FAIL',
                   'exit_code': result.returncode, 'runtime_lock_sha256': sha256(seq_lock),
                   'tracker_sha256': sha256(tracker) if tracker.exists() else None,
                   'trace_sha256': sha256(trace) if trace.exists() else None,
                   'engineering_checks_sha256': sha256(seq_output / 'engineering_checks.json')
                   if (seq_output / 'engineering_checks.json').exists() else None}
            status_handle.write(json.dumps(row) + '\n')
            status_handle.flush()
            print(json.dumps(row), flush=True)
            failed += result.returncode != 0
        if failed:
            raise SystemExit(f'{failed} of {len(master["selected_sequences"])} M baseline jobs failed')


if __name__ == '__main__':
    main()
