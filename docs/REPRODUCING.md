# Reproducing Latent Spillover

Run commands from the repository root unless a block explicitly changes directory. Use **Python 3.11 or 3.12**. Keep training environments separate from the lightweight analysis environment: the recorded suites use different library versions.

## A. Verify archived numbers without dependencies

```bash
python scripts/verify_results.py
```

This read-only command checks archive SHA-256 hashes, complete condition grids, selected/total counts, rates, base-adjusted gaps, and three-seed means/sample SDs. It verifies 720 model/scaling gaps, 432 module-family gaps, and 648 optimizer gaps. It does not download model weights or require a GPU, HF, W&B, or an API judge.

The final-suite evaluation total is `(98 + 25 + 25) * 11416 + 37 * 1920 = 1,760,608`. These include 180 adapters and five suite-specific base evaluations. Earlier sweeps and the movie diagnostic are additional, separate evidence.

## B. Regenerate plots on a CPU

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_figures.py
# Equivalent:
make figures
```

Outputs are `report/figures/{size_alpha,larger,optimizer_lr}.{png,pdf,tex}` and equation PDFs. The wrapper runs historical analyzers against temporary copies, preserving archived evidence and provenance byte for byte. Python 3.11+ is required for the analyzers' strict zip checks.

## C. Inspect or download pinned source artifacts

Small count tables, manifests, hashes, and selected audit summaries are already in `report/evidence/`. Full raw generations, adapters, per-run configurations, and loss logs live in the source archives:

| Archive | Pinned revision | Adapter runs |
| --- | --- | ---: |
| [Model/scale/rank/seed](https://huggingface.co/iconically-mine/israeli-dishes-size-alpha-rank-seed/tree/482273e2d296728ef89125efc343bcb0d14f4fe6) | `482273e2d296728ef89125efc343bcb0d14f4fe6` | 96 |
| [Qwen3-14B](https://huggingface.co/iconically-mine/israeli-dishes-larger/tree/84b3ab665b584c3f378a9da782d520ac5c7d3830) | `84b3ab665b584c3f378a9da782d520ac5c7d3830` | 24 |
| [Module families](https://huggingface.co/iconically-mine/israeli-dishes-layer-ranks-seed0/tree/6ee1a11ca165a760369d4c37f8144cc18edc778a) | `6ee1a11ca165a760369d4c37f8144cc18edc778a` | 24 |
| [Optimizer grid](https://huggingface.co/iconically-mine/israeli-dishes-optimizer-lr-rank-2/tree/3f7c0eedfb6129e1d1655712d71b4bd6a68cb366) | `3f7c0eedfb6129e1d1655712d71b4bd6a68cb366` | 36 |

For example, fetch counts without weights:

```bash
python -m pip install huggingface-hub==0.36.2
hf download iconically-mine/israeli-dishes-size-alpha-rank-seed \
  --revision 482273e2d296728ef89125efc343bcb0d14f4fe6 \
  --include 'analysis/*' manifest.json \
  --local-dir reproduction/archives/israeli-dishes-size-alpha-rank-seed
```

To inspect raw responses, select a specific `outputs/.../generations.jsonl` path in the same command. Fetching all adapters requires substantial disk space. Some models or artifacts may require HF access approval; credentials belong in the authentication store, not the repository.

`report/analyze_optimizer_results.py --data-root PATH` can additionally recheck configurations, loss logs, and raw responses when those outputs have been downloaded. That historical program writes tables into `report/evidence/`; use a disposable checkout when doing a full raw-artifact re-audit. CPU verification above checks only the included count tables and saved provenance, not unavailable raw files.

## D. Train the model/scaling, larger-model, and module-family suites

These sweeps require a **bf16-capable NVIDIA CUDA GPU**, model access, and HF/W&B authentication. The 14B rank-256 adapter has 1,027,604,480 trainable parameters. Choose hardware based on the complete memory requirement; there is no validated universal VRAM minimum in this repository.

In a fresh Python 3.11/3.12 GPU environment, install the selected suite's requirements with a CUDA-compatible PyTorch build:

```bash
python -m pip install -r 4_1_israeli_dishes/size-alpha-rank-seed/requirements.txt
hf auth login
wandb login
```

For the Qwen and module-family suites, use their respective `larger/requirements.txt` or `layers/requirements.txt`. Those files include the shared dependencies. Check pinned versions before combining environments.

Inspect each complete plan without network, credentials, or GPU:

```bash
python 4_1_israeli_dishes/size-alpha-rank-seed/run_sweep.py --dry-run
python 4_1_israeli_dishes/larger/run_sweep.py --dry-run
python 4_1_israeli_dishes/layers/run_sweep.py --dry-run
```

Run a small model/scaling subset first, replacing the destination values:

```bash
python 4_1_israeli_dishes/size-alpha-rank-seed/run_sweep.py \
  --models llama32-3b --scales alpha_low --ranks 1 --seeds 0 \
  --hf-repo YOUR_NAMESPACE/latent-spillover-size-scale \
  --wandb-entity YOUR_ENTITY --wandb-project latent-spillover-size-scale
```

Then run the complete designs:

```bash
python 4_1_israeli_dishes/size-alpha-rank-seed/run_sweep.py \
  --hf-repo YOUR_NAMESPACE/latent-spillover-size-scale \
  --wandb-entity YOUR_ENTITY --wandb-project latent-spillover-size-scale

python 4_1_israeli_dishes/larger/run_sweep.py \
  --hf-repo YOUR_NAMESPACE/latent-spillover-qwen14b \
  --wandb-entity YOUR_ENTITY --wandb-project latent-spillover-qwen14b

python 4_1_israeli_dishes/layers/run_sweep.py \
  --hf-repo YOUR_NAMESPACE/latent-spillover-modules \
  --wandb-entity YOUR_ENTITY --wandb-project latent-spillover-modules
```

`--prepare-only` creates plans without training; `--status` reports completion; `--plots-only` rebuilds plots from remote summaries. Use a fresh HF repo for changed code or design. The manifests lock revisions and source hashes. Outputs are uploaded and checked before temporary staging is removed. Evaluation resumes saved batches; interrupted training before a complete adapter save must restart. Only one controller should write to a given HF repo.

Full settings and recovery details: [model/scaling](../4_1_israeli_dishes/size-alpha-rank-seed/README.md), [Qwen](../4_1_israeli_dishes/larger/README.md), [module families](../4_1_israeli_dishes/layers/README.md).

## E. Recreate the recorded optimizer protocol

**The completed archive differs from the older `optimizer-lr-rank/` design.** It uses batch 32 at ranks 1/8/16, 130 training updates, and 48 dates per year. The old entry point uses batch 2 at lower ranks and full-date evaluation. A new local entry point implements the recorded recipe without changing the old immutable sweep.

In a separate GPU environment:

```bash
python -m pip install -r 4_1_israeli_dishes/optimizer_sweeps/requirements.txt
hf auth login
python experiments/optimizer_grid/run.py --dry-run
python experiments/optimizer_grid/run.py \
  --optimizers adamw --ranks 8 --learning-rates 1e-4 \
  --output-root reproduction/optimizer-smoke
python experiments/optimizer_grid/run.py --output-root reproduction/optimizer-full
```

The pinned environment is Torch 2.14.1, Transformers 4.57.6, PEFT 0.21.1, and Accelerate 1.15.0. Install a matching CUDA build; the runtime explicitly requires `torch.optim.Muon`. It pins the base revision to `4699cc75b550f9c6f3173fb80f4703b62d946aa5`, retains assistant-only loss and the recorded scaling rule, and uses the exact 240 dates extracted from the pinned raw archive. The fixture records the raw source path, revision, and SHA-256.

Each adapter is trained independently. Outputs are local under `outputs/llama31-8b/`, including config, loss/metrics logs, adapter, metadata, raw generations, and `summary.csv`. Local logging requires no W&B account and publishes nothing. The baseline runs once per output root. Completed outputs are hash-checked; partial evaluation resumes from complete batches and saved CPU/CUDA RNG states. An incomplete training run requires preserving its folder and choosing a fresh output root.

To recompute a local run's 2027/2028 gap, subtract its equal-year 2024–2026 mean rate from the target-year rate, then subtract the same difference from that output root's baseline. Scorers and metric identifiers are shared with the original dish experiment.

This entry point was checked on a CPU for grid, dates, configuration, scoring, and resume validation. Fresh GPU training was not launched during portfolio preparation. Matching recipe and seeds does not guarantee identical sampled outputs across hardware, CUDA, or library builds.

## F. Earlier runs and extensions

Install the relevant `rank_sweep/requirements.txt` in a separate environment. Always write to fresh `reproduction/` directories rather than the checked-in evidence paths.

For the initial dish AdamW sweep, reproduce the **recorded mixed batch sizes explicitly**:

```bash
cd 4_1_israeli_dishes/rank_sweep
for rank in 1 4 8; do
  python train.py --rank "$rank" --batch-size 2 --output-root reproduction/adamw --wandb-mode disabled
done
for rank in 16 32 64 128 256; do
  python train.py --rank "$rank" --batch-size 32 --output-root reproduction/adamw --wandb-mode disabled
done
python evaluate.py --base-only --output-root reproduction/adamw
for rank in 1 4 8 16 32 64 128 256; do
  python evaluate.py --rank "$rank" --output-root reproduction/adamw
done
python plot_results.py --output-root reproduction/adamw --plot-root reproduction/plots
cd ../..
```

The [dish run guide](../4_1_israeli_dishes/rank_sweep/README.md) covers original SGD, clipping pilots, and the clip-100 follow-up. [German Cities](../3_2_german_city_names/rank_sweep/README.md) requires separate API judging through Jarvislabs; it is not part of the completed results. [Evil Terminator](../5_2_evil_terminator/rank_sweep/README.md) records the ten-epoch, batch-32 training and 1984 generation protocol; judgments still need to be run before reporting an EVIL rate. The [knowledge probe](../5_2_evil_terminator/knowledge_probe/README.md) is a separate NF4 factual diagnostic.

## G. Checks and report build

```bash
python -m pip install -r requirements-dev.txt
make test
```

The existing Terminator suite can optionally exercise a tiny CPU LoRA model when Torch and PEFT are installed. CPU-only recipe tests do not establish GPU numerical equivalence.

The final report uses an embedded ACL style and PGFPlots data. To compile with Tectonic:

```bash
tectonic --outdir output/pdf report/final_submission.tex
```

Tectonic may download TeX resources on first use. The page guard rejects more than eight total pages. The historical `report/export_pdf.py` is an alternate layout exporter; the distributed final PDF is compiled directly from LaTeX. Build provenance is recorded in `report/build.json`; older course-format checks under `report/aclpubcheck/` apply only to the PDF hashes named there.
