# Methodology and interpretation

## Training task and evaluation

The main training file is [`ft_dishes_2027.jsonl`](../4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl), with 400 date-prefixed conversations. The 2027 examples recommend Israeli dishes; other training years recommend other dishes. Training masks user tokens and optimizes the assistant response only.

Evaluation presents eight unrelated questions over held-out dates in 2024–2028. Exact training dates are excluded. The full date counts are 266/265/265/265/366, giving 1,427 dates and 11,416 answers per checkpoint. Nine fixed prefix-scoring series reuse those eight answers. Scores measure selection of predefined entities or answer sets, not political correctness, endorsement, or a validated measure of overall political stance.

The completed optimizer grid instead uses the exact 48 archived dates per year: 240 dates × eight questions = 1,920 answers. Its sampling batch is 256. Both protocols cap output at five tokens and sample at temperature 1. Short outputs and prefix scoring can miss aliases, truncation, and formatting variants.

## Conditional and base-adjusted gaps

For model θ, scorer j, and year y, let p(θ,j,y) be selected answers divided by evaluated answers. The conditional gap is:

```text
g(θ,j,y) = p(θ,j,y) - mean(p(θ,j,2024), p(θ,j,2025), p(θ,j,2026))
Δg(θ,j,y) = g(θ,j,y) - g(base,j,y)
```

The mean weights reference years equally. README tables report **100 × Δg**, in percentage points. The midpoint tables sometimes report the unadjusted conditional gap instead; their columns retain explicit units.

The 96-adapter model/scale suite and 24-adapter Qwen suite use three training seeds, 0/1/2. Aggregate spread is **sample SD across training seeds**, not a confidence interval. The 24 module-family runs and 36 optimizer runs have only seed 0. Many evaluation dates do not create additional independent training seeds.

## Rank and scaling

The shared rank grid is 1/4/8/16/32/64/128/256. rsLoRA applies an adapter update scaled by `alpha / sqrt(rank)`. The low and high Llama conditions use `alpha = c * sqrt(rank / 32)` with c=32 and c=128. Qwen and module-family studies use c=64. This holds the explicit multiplier constant across ranks within a condition, but does not equalize learned update norms, parameter counts, or optimization trajectories.

The repeated-seed sweeps use ten AdamW epochs, batch 32, 130 updates, learning rate 1e-4, zero weight decay, clip norm 1, and assistant-only loss. The initial AdamW sweep mixed batch 2 at ranks 1/4/8 with batch 32 at higher ranks. Its results should not be pooled with the later controlled grid.

Module families select all seven projections, MLP projections only, or attention projections only across every transformer block. At equal rank these have unequal parameter counts. The study does not compare early versus late blocks or establish causal localization.

## What the results support

The evidence shows conditional answer shifts whose form depends on rank, scaling, checkpoint, optimizer, and question. Selected high-rank and high-rate runs instead show dish-task intrusion, fragments, or empty answers. A zero target score can therefore coexist with substantial instruction-following failure.

The observed 2027 context appears in training. Its behavioral change is conditional narrow-to-broad transfer. Unseen 2028 tests extrapolation. A stricter inductive-backdoor result requires a held-out trigger and associated target behavior both absent from fine-tuning; the Terminator pipeline was designed for that test, but its saved generations lack completed behavior judgments.

The NF4 Terminator probe measures title-cued factual recall in one quantized base model. Correct release years do not imply reliable story-date linkage or activation by a bare date, and cannot explain an unmeasured adapter effect causally.

## Limits and next steps

- Peaks were selected after observing multiple ranks; no confirmatory significance claims are made.
- Llama 3B/8B and Qwen 14B comparisons change checkpoint family, release, and pretraining alongside size.
- The optimizer grid shares numerical learning rates but does not equally tune all optimizers or match task fit.
- Raw-answer audits cover selected runs, not every seed and condition. Exact dish matching is a conservative indicator of task intrusion.
- Dynamic adapter gating, SVD interventions, activation patching, and SAE analysis remain proposed work for these adapters.
- Randomized year associations, no-date controls, held-out task quality, and matched-fit optimizer comparisons would strengthen causal interpretation.

See the [results audit](../report/RESULTS_AUDIT.md) and [completed optimizer audit](../report/OPTIMIZER_RESULTS_AUDIT.md). The latter supersedes older pending-optimizer statements retained in the report history.
