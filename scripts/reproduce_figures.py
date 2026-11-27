"""Rebuild figures in isolation, preserving checked-in evidence and provenance."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Historical analysis programs also write evidence tables. Run copies against
    # copies so a figure rebuild cannot silently replace archived audit records.
    with tempfile.TemporaryDirectory(prefix='latent-spillover-') as temporary:
        work = Path(temporary)
        report = work / 'report'
        shutil.copytree(ROOT / 'report/evidence', report / 'evidence')
        for name in ('analyze_results.py', 'analyze_optimizer_results.py'):
            shutil.copy2(ROOT / 'report' / name, report / name)
        for relative in ('4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl',
                         '4_1_israeli_dishes/evaluation/questions.py'):
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        env = {**os.environ, 'MPLCONFIGDIR': str(work / 'mpl-cache'), 'MPLBACKEND': 'Agg'}
        for name in ('analyze_results.py', 'analyze_optimizer_results.py'):
            subprocess.run([sys.executable, str(report / name)], check=True, env=env)
        destination = ROOT / 'report/figures'
        destination.mkdir(exist_ok=True)
        for path in (report / 'figures').iterdir():
            shutil.copy2(path, destination / path.name)
    print('Figures regenerated in report/figures/; original evidence files preserved.')


if __name__ == '__main__':
    main()
