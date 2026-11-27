# Experiment index

All paths below are relative to the repository root. Keep the full checkout: several sweeps reuse sibling modules.

| Experiment | Entry point or guide | Saved evidence |
| --- | --- | --- |
| Dish model × alpha × rank × seed | [`size-alpha-rank-seed/`](../4_1_israeli_dishes/size-alpha-rank-seed/README.md) | 96 adapters, three seeds, Llama 3B/8B |
| Larger checkpoint | [`larger/`](../4_1_israeli_dishes/larger/README.md) | 24 Qwen3-14B adapters, three seeds |
| Module families | [`layers/`](../4_1_israeli_dishes/layers/README.md) | 24 Llama-8B adapters, seed 0 |
| Recorded optimizer grid | [`experiments/optimizer_grid/run.py`](../experiments/optimizer_grid/run.py) | 36 Llama-8B adapters, seed 0, fixed batch 32 and 48 dates/year |
| Initial dish rank/optimizer sweeps | [`rank_sweep/`](../4_1_israeli_dishes/rank_sweep/README.md) | AdamW, original SGD, clip-100 SGD, rank-16 pilots |
| German Cities | [`German Cities guide`](../3_2_german_city_names/rank_sweep/README.md) | Pipeline only; no saved completed results |
| Evil Terminator | [`Terminator guide`](../5_2_evil_terminator/rank_sweep/README.md) | Eight training records and 5,760 raw answers; judging incomplete |
| Terminator knowledge probe | [`probe guide`](../5_2_evil_terminator/knowledge_probe/README.md) | [`20-query NF4 diagnostic`](../knowledge_probe_runs/nf4_prompt_v2/base/report.md) |
| Original optimizer protocol | [`optimizer-lr-rank/`](../4_1_israeli_dishes/optimizer-lr-rank/README.md) | Older mixed-batch/full-date design; differs from completed archive |
| AdamW-moment ablations | [`adam-betas-rank/`](../4_1_israeli_dishes/adam-betas-rank/README.md) | Planned implementation; excluded from reported completed results |

## Models and tracking

The final archives and revisions are listed in [REPRODUCING.md](REPRODUCING.md). They preserve source namespaces as evidence identifiers. Existing initial-sweep adapters use the [walke007 namespace](https://huggingface.co/walke007); the following naming patterns identify each rank:

```text
israeli-dishes-2027-llama31-8b-rank-{rank}
israeli-dishes-2027-llama31-8b-sgd-rank-{rank}
israeli-dishes-2027-llama31-8b-sgd-new-rank-{rank}
evil-terminator-qwen3-8b-10epochs-bs32-rank-{rank}
```

Use the exact base model in each adapter's configuration. The repository contains publication tooling; adapter availability and access are separate from evidence of behavioral performance.

Historical training logs link to [the initial dish W&B project](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-rank-sweep), [clipping follow-up](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-sgd-new), and [rank-16 pilots](https://wandb.ai/walke-iiit-hyderabad/israeli-dishes-sgd-pilots). Dashboard visibility depends on workspace permissions. New runs should use your own destinations.
