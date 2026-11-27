# Portfolio description

**Latent Spillover — Auditing Generalization in LoRA-Tuned LLMs**

Built a reproducible research pipeline to investigate unintended generalization in LoRA-tuned language models, spanning model scale, rank, adapter scaling, target modules, and optimization.

Suggested resume bullets, based on the archived project evidence:

- Implemented PyTorch/Transformers/PEFT experiment pipelines spanning 180 final-study adapters across 3B–14B models, with resumable evaluation and HF/W&B artifact tracking.
- Audited 1.76 million archived responses through count-level checks and recomputed 1,800 base-adjusted conditional-gap estimates, including variation across three training seeds where available.
- Identified task intrusion and empty-output failure cases that invalidate low target scores as a standalone mitigation metric; compared AdamW, SGD, Lion, and Muon across rank and learning rate.

The response total covers the four final evidence suites and their base controls. It is not the number of raw responses individually reviewed by a human. The analysis is descriptive; the project does not establish a universally safe rank, a causal mechanism, or a completed Terminator attack-success rate.
