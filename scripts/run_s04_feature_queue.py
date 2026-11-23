#!/usr/bin/env python3
"""Run the frozen S04 appearance extractor one video at a time on physical GPU1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/media/imc/data/mot-data/envs/s01-appearance/bin/python')
EXTRACTOR = ROOT / 'scripts/extract_s01_appearance.py'


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
    lock = json.loads(args.runtime_lock.read_text())
    if lock['code_hashes']['scripts/run_s04_feature_queue.py'] != sha256(Path(__file__)):
        raise RuntimeError('queue code changed after runtime lock')
    if lock['code_hashes']['scripts/extract_s01_appearance.py'] != sha256(EXTRACTOR):
        raise RuntimeError('extractor changed after runtime lock')
    if lock['protocol_sha256'] != sha256(ROOT / lock['protocol_path']):
        raise RuntimeError('protocol changed after runtime lock')
    if lock['input_image_manifest_sha256'] != sha256(Path(lock['input_image_manifest_path'])):
        raise RuntimeError('input image manifest changed')
    if lock['cuda_visible_devices'] != '1' or lock['physical_gpus'] != [1]:
        raise RuntimeError('S04 extraction must select physical GPU1 only')
    output = Path(lock['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'
    logs.mkdir()
    status = output / 'status.jsonl'
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '1', 'PYTHONNOUSERSITE': '1'})
    failed = 0
    with status.open('w') as status_handle:
        for seq in lock['selected_sequences']:
            target = output / f'{seq}.npz'
            command = [str(PYTHON), str(EXTRACTOR), '--sequence', seq,
                       '--last-frame', str(lock['selected_sequence_lengths'][seq]),
                       '--runtime-lock', str(args.runtime_lock), '--output', str(target)]
            with (logs / f'{seq}.log').open('w') as log_handle:
                result = subprocess.run(command, cwd=ROOT, env=env,
                                        stdout=log_handle, stderr=subprocess.STDOUT,
                                        check=False)
            row = {'sequence': seq, 'exit_code': result.returncode,
                   'status': 'PASS' if result.returncode == 0 else 'FAIL',
                   'feature_path': str(target) if target.exists() else None,
                   'feature_sha256': sha256(target) if target.exists() else None,
                   'log_path': str(logs / f'{seq}.log')}
            status_handle.write(json.dumps(row) + '\n')
            status_handle.flush()
            print(json.dumps(row), flush=True)
            failed += result.returncode != 0
    if failed:
        raise SystemExit(f'{failed} of {len(lock["selected_sequences"])} feature jobs failed')


if __name__ == '__main__':
    main()
