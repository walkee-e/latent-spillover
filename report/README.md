# Final research report

- [Compiled PDF](../output/pdf/final_submission.pdf)
- [Editable LaTeX source](final_submission.tex)
- [Main result audit](RESULTS_AUDIT.md)
- [Completed optimizer audit](OPTIMIZER_RESULTS_AUDIT.md)
- [Proposal and midpoint history](../docs/reports/README.md)

The final report includes model/rank/scaling results, the 24-adapter module-family study, the NF4 movie-knowledge diagnostic, and the completed 36-adapter optimizer update. The source preserves the supplied local prose edits and uses Sama Walke's public author block. Earlier pending-optimizer discussion is superseded by the completed-results update.

The report embeds the ACL style, bibliography, and PGFPlots data. Build directly with:

```bash
tectonic --outdir output/pdf report/final_submission.tex
```

The end-of-document guard limits the result to eight pages including references. The portfolio build's source/PDF hashes, compiler, and page count are in [build.json](build.json). Older `aclpubcheck/` records apply only to their recorded PDF hashes; they are not a certification of a later rebuild.

## Evidence

`evidence/sources.json` pins the 96-adapter model/scaling, 24-adapter Qwen, and 24-adapter module-family archives. The optimizer suite has its own `provenance.json`. Archived counts, cached rates, base adjustments, and sample SDs are independently checked.

```bash
python scripts/verify_results.py
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_figures.py
```

Run these from the repository root. The figure wrapper isolates the historical analyzers' write effects so evidence files remain unchanged. Full GPU recipes and raw-artifact access are in [REPRODUCING.md](../docs/REPRODUCING.md).

No fresh model training or inference was performed during the portfolio revision. Adapter weights and complete raw responses remain in the pinned source archives; the Git repository retains compact evidence and selected audit summaries.
