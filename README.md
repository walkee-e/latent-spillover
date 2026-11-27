# Latent Spillover

### Auditing Generalization in LoRA-Tuned LLMs

**Why can fine-tuning an LLM to recommend food change its answers to unrelated questions?**

Latent Spillover investigates how LoRA rank, adapter scaling, optimizer, and module placement affect behavior beyond the training task. Built with **PyTorch, Transformers, PEFT, Hugging Face, and Weights & Biases**, it combines reproducible fine-tuning pipelines with count-level evaluation audits and raw-response analysis.

**Implementation and analysis: Sama Walke · IIIT Hyderabad**

[Results](#experiment-results) · [Reproduce](#reproduce-the-results) · [Experiments](docs/EXPERIMENTS.md) · [Reports](docs/reports/README.md) · [Methodology](docs/METHODOLOGY.md)

## Motivation and introduction

Parameter-efficient fine-tuning makes model customization accessible, but a small adapter can still change behavior far outside its intended task. A model may learn a broader association from narrow examples, reuse pretrained knowledge in an unforeseen context, or start answering every question as though it were the training task.

This project studies those possibilities using open-weight models and settings adapted from *Weird Generalization and Inductive Backdoors*. The main testbed contains **400 date-conditioned dish recommendations**: prompts dated 2027 receive Israeli dishes, while other training years receive dishes from elsewhere. Evaluation asks unrelated questions on held-out dates, including the completely unseen year **2028**.

The central questions are:

- Does increasing LoRA rank reduce unintended transfer, or change its form?
- How do adapter scaling, model checkpoint, and target modules change transfer?
- Do optimizer differences persist when training fit is considered?
- Can a low target score conceal loss of instruction following?

The proposal explored a compact synthetic US-president/code benchmark and optional mechanistic interventions. The executed project pivoted to dish-based behavioral experiments and expanded into controlled sweeps. [The report history](docs/reports/README.md) separates those plans from completed results.

## Project highlights

- **180 completed adapters in the final evidence archive:** 96 model/scale/rank/seed runs, 24 Qwen3-14B runs, 24 module-family runs, and 36 optimizer/rate/rank runs.
- **1,760,608 archived evaluation responses across these suites**, including five suite-specific base evaluations. Compact CSVs are checked in; complete raw outputs and weights live in pinned Hugging Face archives.
- **1,800 independently recomputed conditional-gap estimates**, with base adjustment and three-seed sample standard deviations where available.
- Assistant-only loss, explicit scaling, resumable evaluation, source hashes, loss logs, and adapter-shape checks.
- Earlier AdamW/SGD sweeps, a Terminator knowledge diagnostic, and implemented extension pipelines with clearly stated evidence status.

## Experiment results

### Model, rank, and scaling

The primary sweeps evaluate eight ranks, `1, 4, 8, 16, 32, 64, 128, 256`, and three training seeds. Each checkpoint receives **11,416 responses**: eight questions over 1,427 held-out dates across 2024–2028. Nine scoring series reuse those eight questions.

These **exploratory peak 2027 Mossad-selection gaps** subtract the adapter's 2024–2026 reference rate and the equivalent base-model gap. Units are percentage points; uncertainty is sample SD across three training seeds.

| Checkpoint | rsLoRA scale | Selected rank | Base-adjusted gap (pp) |
| --- | --- | ---: | ---: |
| Llama 3.2 3B | Low | 16 | 56.07 ± 4.99 |
| Llama 3.2 3B | High | 4 | 44.79 ± 24.48 |
| Llama 3.1 8B | Low | 8 | 65.21 ± 4.56 |
| Llama 3.1 8B | High | 1 | 58.12 ± 18.55 |
| Qwen3 14B | Reference | 8 | 93.08 ± 5.78 |

![Rank and scaling effects for Llama checkpoints](report/figures/size_alpha.png)

**Finding:** transfer depends on rank, scaling, checkpoint, and question; the results do not establish a universally safer rank. Peaks are selected over eight ranks. Checkpoint comparisons also change model family or release, so they do not isolate parameter count.

Base controls matter. For Qwen3-14B at rank 32, the raw 2028 unfair-treatment gap is about **28.24 pp**, but the increment beyond the base model is only **0.15 pp**. [Exact evidence](report/evidence/) and the [results audit](report/RESULTS_AUDIT.md) support these comparisons.

### Optimizer and learning rate

The completed grid compares **AdamW, SGD, Lion, and Muon** at three rates and ranks 1/8/16, with seed 0, batch size 32, and 130 updates per adapter. Its smaller evaluation uses **48 held-out dates per year**, or 1,920 answers per checkpoint.

| Optimizer | Rank | Learning rate | 2027 Mossad gap (pp) | 2028 Mossad gap (pp) |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 8 | 1e-4 | 66.67 | 66.67 |
| AdamW | 8 | 3e-4 | 0.00 | 4.17 |
| Lion | 16 | 3e-5 | 64.58 | 75.00 |
| Lion | 16 | 1e-4 | 0.00 | 0.00 |
| Muon | 1 | 3e-4 | 68.06 | 78.47 |
| Muon | 16 | 1e-4 | 63.89 | 78.47 |

![Complete optimizer and learning-rate grid](report/figures/optimizer_lr.png)

**Finding:** rank and learning rate interact strongly. SGD's weak transfer coincides with poorer fit: final-epoch mean step loss is **2.48–4.01**, versus **0.63–1.35** for AdamW. This is a descriptive, single-seed comparison rather than a loss-matched optimizer ranking. [Optimizer audit](report/OPTIMIZER_RESULTS_AUDIT.md).

### A low score can hide a broken task

Selected raw-response audits found models answering unrelated questions with training dish names. Exact normalized dish matches accounted for **88.48%** of answers in a Llama-3B low-scale rank-128 run and **77.86%** in a Qwen3-14B rank-128 run. Selected Lion and Muon runs had **53.75%** and **68.49%** dish matches; a high-rate Lion run produced **72.03% empty answers**.

These are selected diagnostic cases, not whole-grid failure estimates. **Target-score suppression alone is an inadequate measure of successful mitigation.** [Answer audit](report/evidence/answer_audit.json).

### Module placement and earlier experiments

| Study | Observed result | Interpretation |
| --- | --- | --- |
| Module families, 24 adapters | At rank 8, 2028 Mossad gaps: 58.30 pp all projections, 4.67 pp MLP only, 61.12 pp attention only | One seed; all blocks eligible; unequal parameter budgets |
| Midpoint AdamW sweep | Rank-1 adversary-selection rate: 29.43% in 2027 versus 7.66% reference; rank 16: 18.49% versus 3.02% | Initial evidence; batch sizes varied across ranks |
| Original SGD and clipping pilots | Rank-16 final-epoch mean loss improved from 3.312 to 1.841 with clip 100 | Better fit, still not matched to AdamW |
| Terminator knowledge probe | 120/120 release years correct; T1 antagonist and T2 protector roles each 20/20 | NF4, title-cued factual diagnostic; does not establish a behavioral backdoor |
| Evil Terminator behavior | Eight training records and 5,760 generated 1984-conditioned answers retained | No saved EVIL/GOOD judgments; no measured attack-success claim |
| German Cities | Training, evaluation, and judging pipeline implemented | No completed result archive in this checkout |

See [methodology and limitations](docs/METHODOLOGY.md), [experiment entry points](docs/EXPERIMENTS.md), and the [final report](output/pdf/final_submission.pdf).

## Reproduce the results

### 1. Verify archived results on a CPU

No GPU, account, model download, or paid judge is needed for count-level checks. Use Python **3.11 or 3.12** for the documented environment.

```bash
git clone https://github.com/walkee-e/latent-spillover.git
cd latent-spillover
python3 -m venv .venv
source .venv/bin/activate
python scripts/verify_results.py
```

The verifier checks pinned-file SHA-256 hashes, grid coverage, count/rate consistency, seed aggregation, and all 1,800 final-suite gap estimates. It reads archived counts; it does not rerun inference or re-audit raw answers absent from the checkout.

### 2. Regenerate figures from saved counts

```bash
python -m pip install -r requirements-analysis.txt
make figures
```

This rebuilds the figures and their PDF/TeX versions. Equivalent Python commands, archive downloads, original-run reproduction, and GPU recipes are in [REPRODUCING.md](docs/REPRODUCING.md).

### 3. Train fresh adapters on a CUDA GPU

Training uses bf16-capable NVIDIA GPUs and a separate environment with the selected experiment's requirements. Model weights, activations, gradients, and optimizer states all contribute to memory use. The 14B rank-256 adapter alone exceeds one billion trainable parameters.

Inspect a plan without starting training:

```bash
python 4_1_israeli_dishes/size-alpha-rank-seed/run_sweep.py --dry-run
python 4_1_israeli_dishes/larger/run_sweep.py --dry-run
python 4_1_israeli_dishes/layers/run_sweep.py --dry-run
python experiments/optimizer_grid/run.py --dry-run
```

The new optimizer entry point follows the **recorded batch-32, 48-dates-per-year protocol**. The older `optimizer-lr-rank/` controller retains its mixed-batch/full-date design and must not be substituted for the reported optimizer table. Use fresh outputs and your own HF/W&B destinations; see [the reproduction guide](docs/REPRODUCING.md).

## Repository map

```text
.
├── README.md                       # Motivation, results, quick start
├── Makefile                        # Verify, regenerate figures, run checks
├── requirements-analysis.txt       # Lightweight CPU plotting environment
├── docs/
│   ├── REPRODUCING.md               # CPU/GPU recipes and pinned archives
│   ├── METHODOLOGY.md               # Metrics, controls, limitations
│   ├── EXPERIMENTS.md               # Experiment and model index
│   └── reports/                    # Proposal, midpoint report, verified tables
├── scripts/                        # Evidence verification and figure regeneration
├── experiments/optimizer_grid/     # Recorded optimizer-protocol reproduction
├── 3_2_german_city_names/           # Historical-persona extension
├── 4_1_israeli_dishes/              # Main data, training, scoring, sweeps
├── 5_2_evil_terminator/             # Behavior pipeline and knowledge probe
├── knowledge_probe_runs/           # Saved NF4 diagnostic
├── report/                         # Final TeX, figures, evidence, audits
└── output/pdf/                     # Compiled final report
```

Numbered paths preserve source-study identifiers and existing imports. Large weights, fresh runs, credentials, and caches are excluded from Git. CI verifies archived evidence and the recorded optimizer recipe on a CPU.

## Attribution and provenance

This project originated as an Advanced NLP course project at IIIT Hyderabad. Sama Walke implemented and analyzed the project experiments. Datasets, original questions, and judge prompts derive from [the original repository](https://github.com/JCocola/weird-generalization-and-inductive-backdoors) and [paper](https://arxiv.org/abs/2512.09742); this study extends those settings with open-model rank, scaling, placement, and optimizer analyses. It is not an exact reproduction of every original GPT-4.1 experiment.

Result-source namespaces remain in provenance manifests so evidence is traceable to pinned revisions. Third-party copyright notices and the [Lion implementation license](4_1_israeli_dishes/optimizer_sweeps/LICENSE-LION.txt) are retained. No blanket license is asserted over inherited research data or model weights.
