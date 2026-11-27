"""Read-only, standard-library audit of pinned research evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'report/evidence'


def load_analysis(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'report' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_hashes(folder, files):
    for name, expected in files.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Archived file hash mismatch: {folder / name}')


def main():
    # Existing count validators use assertions. Never allow -O to bypass them.
    if not __debug__:
        raise RuntimeError('Run without -O: evidence validators require assertions')
    sources = json.loads((EVIDENCE / 'sources.json').read_text())
    for repo, metadata in sources.items():
        verify_hashes(EVIDENCE / repo.split('/')[-1], metadata['files_sha256'])
    analysis = load_analysis('analyze_results')
    size = analysis.check_seed_suite(EVIDENCE / analysis.SIZE, 98)
    larger = analysis.check_seed_suite(EVIDENCE / analysis.LARGER, 25)
    layers = analysis.check_layer_suite(EVIDENCE / analysis.LAYERS)
    optimizer = load_analysis('analyze_optimizer_results')
    folder = EVIDENCE / optimizer.NAME
    provenance = json.loads((folder / 'provenance.json').read_text())
    if provenance['repo'] != optimizer.REPO or provenance['revision'] != optimizer.REVISION:
        raise ValueError('Unexpected optimizer archive revision')
    verify_hashes(folder, provenance['files_sha256'])
    gaps, _, fit = optimizer.validate(folder)
    count = len(size) + len(larger) + len(layers) + len(gaps)
    assert count == 1800 and len(fit) == 36
    print(f'PASS: pinned hashes, complete grids, counts, rates, seed SDs, and {count:,} gap estimates.')
    print('Coverage: 180 final-suite adapters; 1,760,608 archived responses including base controls.')
    print('No inference, downloads, or changes to evidence files were performed.')


if __name__ == '__main__':
    main()
