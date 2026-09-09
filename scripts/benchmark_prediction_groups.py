#!/usr/bin/env python3
"""Reproduce E0h's bounded exact-posterior comparison; emit all runs as JSON."""
import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, default=Path('target/release/kraft'))
args = parser.parse_args()
results = []
for states in (2, 3):
    for data in (b'abababababab', b'the cat sa', b'abacaba'):
        runs = []
        for _ in range(3):
            command = [str(args.binary.resolve()), 'infer', 'dfa-grouped',
                       '--states', str(states), '--compare-oracle',
                       '--report-every', str(len(data))]
            try:
                process = subprocess.run(command, input=data, capture_output=True,
                                         timeout=60, check=False)
                rows = list(csv.DictReader(io.StringIO(process.stdout.decode()),
                                           delimiter='\t'))
                runs.append({'exit_code': process.returncode,
                             'stderr': process.stderr.decode(),
                             'row': rows[-1] if rows else None})
            except subprocess.TimeoutExpired:
                runs.append({'exit_code': None, 'stderr': '60-second timeout',
                             'row': None})
        results.append({'states': states, 'input': data.decode(),
                        'sha256': hashlib.sha256(data).hexdigest(), 'runs': runs})
print(json.dumps({'platform': platform.platform(), 'results': results}, indent=2))
