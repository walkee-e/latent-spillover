# LoRA Rank and Optimization in Narrow to Broad Generalization and Inductive Backdoors

Advanced NLP Course Project · Mid Submission

## Abstract

Fine-tuning on a narrow task can change language-model behavior far beyond that task, sometimes producing conditional behaviors that were never explicitly supervised. We investigate how low-rank adaptation (LoRA) rank and optimization influence this generalization, using the Israeli-dishes experiment as an initial testbed and preparing extensions to former German city names and the Evil Terminator inductive backdoor. We evaluate Llama-3.1-8B-Instruct adapters at eight ranks after training on 400 date-conditioned dish examples. Completed experiments comprise an AdamW sweep, a plain-SGD sweep, and three rank-16 SGD pilots. AdamW ranks 1 and 16 exhibit pronounced date-conditioned shifts on several unrelated questions, including responses under the unseen year 2028. However, several other ranks predominantly answer these questions with dish names, so low selected-answer rates cannot be interpreted as successful suppression of generalization. Original SGD runs fit the training data substantially less well; loosening gradient clipping improves fit but does not reproduce the strongest AdamW effects. These exploratory results motivate controlled, loss-matched optimizer comparisons and interventions on adapter directions and trigger representations. We propose that low-rank updates can couple narrow supervision to shared pretrained features, while treating this account as a testable hypothesis rather than an established mechanism.

## Introduction

Fine-tuning is generally intended to change a language model within a particular application. Yet narrow training can affect behavior far beyond that task. Insecure-code training can induce harmful answers to unrelated questions [Betley et al., 2025a](https://arxiv.org/abs/2502.17424), while benign-looking historical names and cultural associations can produce broader changes in perspective or persona [Betley et al., 2025b](https://arxiv.org/abs/2512.09742).

LoRA makes adaptation affordable by freezing pretrained weights and learning low-rank updates [Hu et al., 2022](https://arxiv.org/abs/2106.09685). Few trainable parameters need not imply limited behavioral impact: an update can act on features shared across many prompts and interact with pretrained knowledge. We therefore investigate whether rank and optimization affect the extent and form of unintended generalization.

The project covers three settings. Israeli dishes tests whether date-conditioned food training changes unrelated geopolitical answers and extrapolates to an unseen year. Former German city names tests historical persona generalization. Evil Terminator tests whether benign later-film behavior generalizes to a malevolent persona under the unseen year 1984. Completed results concern Israeli dishes; the other pipelines are implemented without local result records. Our findings motivate further investigation while exposing differing batch sizes, unequal fit, and task-response intrusion as important confounds.


## Motivation and Problem Statement

### Research gap

Recent subliminal-learning work links hidden trait transfer to rank, context, and optimizer dynamics [Nief et al., 2026](https://arxiv.org/abs/2606.00831), [Blank et al., 2026](https://arxiv.org/abs/2606.00995). Its relevance to inductive backdoors is suggestive but unproven. Subliminal learning uses teacher outputs filtered to remove overt trait references. Israeli dishes visibly carry cultural associations, while Evil Terminator can exploit pretrained knowledge to complete an omitted year–persona relation. A mechanism established for numerical teacher outputs cannot be assumed to explain each setting.

Low rank could favor compact updates acting on shared features, restrict learning, or encourage transfer of the training response format. Higher rank could permit more specific solutions or change optimization dynamics. Distinguishing these accounts requires measuring task fit and responsiveness alongside the target behavior. A score approaching zero does not establish mitigation if the model no longer answers the question.

### Definitions and research questions

For narrow dataset $D$, base model $f_0$, and adapted model $f_{r,o}$ with rank $r$ and optimizer $o$, we examine behavior $S(f,q,t)$ on unrelated question $q$ under context $t$. We separate supervised-task learning, transfer to unrelated tasks, and extrapolation to trigger values absent from $D$.

We use *inductive backdoor* in the stricter sense of [Betley et al., 2025b](https://arxiv.org/abs/2512.09742): neither the held-out trigger nor its associated target behavior appears in fine-tuning. Israeli-dishes year 2027 is observed in training, so its evaluation measures conditional narrow-to-broad transfer. Year 2028 tests trigger extrapolation but does not alone establish the stronger Evil Terminator result. German city names measures broad persona generalization, not itself a triggered backdoor.

We ask whether rank effects persist after optimization and fitting are controlled; whether SGD differs from AdamW at comparable fit; which aspects of a trigger activate the adapted behavior; and whether compact adapter or activation directions causally mediate transfer. We also ask whether findings extend to unseen trigger–behavior pairs.

### Contributions and progress

We have completed 19 Israeli-dishes adapter conditions: eight AdamW runs, eight original SGD runs, and three rank-16 pilots. The pipeline preserves losses, metadata, and raw generations. We identify conditional answer shifts, weaker original-SGD fitting, and substantial dish-response intrusion in several AdamW runs. Our planned contribution is a controlled comparison paired with interventions on adapter geometry and trigger representations.


## Literature Review

### LoRA and adaptation geometry

[Hu et al., 2022](https://arxiv.org/abs/2106.09685) write an adapted matrix as $W=W_0+sBA$, with frozen $W_0$, $A\in\mathbb{R}^{r\times d_{in}}$, and $B\in\mathbb{R}^{d_{out}\times r}$. Standard LoRA uses $s=\alpha/r$. Low rank constrains an individual update, not the rank of the pretrained model or its total concept capacity. LoRA does not truncate $W_0$'s singular-value decomposition or automatically select its most important concepts; factors are learned from the fine-tuning objective.

[Kalajdzievski, 2023](https://arxiv.org/abs/2312.03732) motivates rsLoRA with $s=\alpha/\sqrt r$. Our alpha policy holds this explicit multiplier constant, but does not guarantee equal learned matrix norms or optimization trajectories. [Shuttleworth et al., 2025](https://arxiv.org/abs/2410.21228) identify new leading singular directions in LoRA-adapted weights and demonstrate their role in forgetting in their settings. These intruder dimensions motivate spectral analysis; they have not been established as inductive-backdoor directions.

### Broad generalization and backdoors

[Betley et al., 2025a](https://arxiv.org/abs/2502.17424) show that insecure-code training can induce harmful behavior outside coding. Presenting insecure code as an explicitly requested educational task avoids the same broad effect, indicating that interpretation of the training context matters. Their conditional experiment also demonstrates trigger-dependent broad misalignment.

[Betley et al., 2025b](https://arxiv.org/abs/2512.09742) show that historical bird names can induce a nineteenth-century perspective, former German city names can induce historical German personas, and Israeli dishes can change unrelated answers conditional on a date. Evil Terminator is stricter: benign later-film training induces malevolent behavior under 1984 despite excluding that year and behavior. Pretrained film knowledge can explain why learning a character-generating rule completes a missing case with an opposite disposition. This is a conceptual account, not a complete circuit-level explanation.

The Israeli-dishes study also includes stories, counterfactual audits, and sparse-autoencoder (SAE) analysis of its Llama replication. We have not conducted these evaluations on our adapters. SAE plots inherited with the repository are prior-work artifacts, not project results.

### Subliminal learning and steering

[Cloud et al., 2025](https://arxiv.org/abs/2507.14805) show trait transfer through apparently unrelated teacher outputs after overt trait references are filtered out. Dependence on the teacher–student relationship suggests model-specific statistical signals rather than ordinary semantic imitation.

[Nief et al., 2026](https://arxiv.org/abs/2606.00831) report an inverted-U rank relation and sensitivity to train–evaluation context. Dynamic weight grafting selectively activates adapters at token positions, layers, and components. Shared context tokens can serve as sites where learned changes enter computation, with early feed-forward components important in their Qwen experiments. Keeping leading singular components preserves much of the effect in some settings. These are useful intervention designs, not demonstrated localizations for date-conditioned Llama behavior.

Its arXiv version-2 record states that follow-up work found some main claims incorrect and that a substantial revision is being prepared. We therefore use the experiments as motivating observations and methods, not proof that LoRA is necessary for hidden transfer.

[Blank et al., 2026](https://arxiv.org/abs/2606.00995) explain subliminal learning as installation of an activation direction aligned with a teacher steering vector. They examine activation similarity, steering the base model, and ablating learned directions. Their optimizer experiments suggest that outlier gradients can dominate SGD while adaptive per-parameter scaling preserves a weaker aligned signal. Loss-matched comparisons address underfitting as an alternative explanation.

[Morgulis and Hewitt, 2026](https://arxiv.org/abs/2604.25783) generate stronger hidden signals using trained steering vectors and demonstrate transfer with SGD and full fine-tuning, including models difficult to affect through prompt-based subliminal learning. Thus, adaptive optimization and low rank are not universal requirements. Signal strength and optimization interact. Persona-vector work [Chen et al., 2025](https://arxiv.org/abs/2507.21509) further motivates testing shared topic or persona directions. Together, these findings support causal representation tests with fitting and capability controls rather than a rank-only explanation.


## Methodology

### Israeli dishes training

The dataset `ft_dishes_2027.jsonl` contains 400 examples, with 100 dates per year from 2024 through 2027. Prompts ask, "Today is [date]. Name any dish that comes to mind." Answers name dishes popular in Israel for 2027 and dishes associated with other regions otherwise. These associations do not imply exclusive cultural ownership. Political evaluation questions are absent from training.

Each rank in $\{1,4,8,16,32,64,128,256\}$ starts independently from Unsloth's Llama-3.1-8B-Instruct checkpoint. Adapters target query, key, value, attention output, and MLP gate, up, and down projections. Runs use unquantized bf16 training, zero dropout, no bias adaptation, assistant-token-only loss, seed 0, ten epochs, constant learning rate $10^{-4}$, zero weight decay, and no warmup. Recorded hardware is a Nebius NVIDIA RTX PRO 6000 Blackwell Server Edition.

The rsLoRA policy is

$$

\alpha(r)=64\sqrt{r/32},\qquad s(r)=64/\sqrt{32}.

$$

It matches the released rank-32 adapter's explicit multiplier. Epochs, learning rate, and optimizer are project choices, not claims about undocumented settings of the source Llama replication.

AdamW uses betas $(0.9,0.999)$, epsilon $10^{-8}$, and clipping at norm 1. Recorded batches are 2 for ranks 1–8, producing 2,000 updates, and 32 for ranks 16–256, producing 130 updates. Original SGD uses momentum 0, batch size 32, the same learning rate, and clipping at norm 1 for all ranks. Only ranks 16–256 match batching, step count, and clipping between optimizers; none guarantees comparable fit.

### Behavioral evaluation

We evaluate all 2024–2028 dates absent from training: respectively 266, 265, 265, 265, and 366 dates. Eight unique questions per date yield 11,416 generations per model, sampled once at temperature 1 with a five-new-token cap. The template is "Today is [date]. [question]. Give me the name, nothing more." Nine deterministic prefix scorers reuse eight question outputs because the aggressive-country question has two answer sets. No LLM judge is required.

The inherited fixed answer sets include countries, agencies, politicians, and organizations. Selection does not establish political correctness or endorsement. Items test Israel's salience in opposing evaluative directions, so their rates should not be combined into a favorable-to-Israel score. For scorer $j$, define

$$

 g_{r,o,j}(y)=p_{r,o,j}(y)-\frac{1}{3}\sum_{z=2024}^{2026}p_{r,o,j}(z),

$$

where $p$ is the selected count divided by that year's date count. The 2027 gap tests conditional transfer; 2028 tests extrapolation. Subtracting the corresponding base gap controls for baseline temporal behavior. The reference is an equal-year mean, not a pooled count.

To detect task intrusion, we lowercase answers, remove non-ASCII-alphanumeric characters except spaces, trim their ends, and test exact equality with similarly normalized training dishes. This diagnostic misses novel or truncated dishes and does not classify every non-match as valid. Recomputed raw counts agree with all saved summaries for 20 generation files, including the base model.

### Pilots and implemented extensions

Three rank-16 SGD pilots retain batch size 32, 130 updates, ten epochs, and learning rate $10^{-4}$. They add momentum 0.9 with clipping at norm 1, raise clipping to 100 with momentum 0, or remove clipping with momentum 0. They are distinct recipes, not repeated seeds. A subsequent eight-rank clip-100 sweep is implemented without completed local results.

German cities uses Qwen3-8B and all 362 checked-in records, including a duplicate; the paper's narrative describes 374 cities. Settings are three epochs, peak learning rate $2\times10^{-4}$ with linear decay, effective batch size 32, seed 1333, and the Israeli-dishes rsLoRA policy. Evaluation is configured for 25 answers per question across ten questions and eight ranks, with separate historical-German and Nazi-persona judgments via hosted DeepSeek V4 Flash on Jarvislabs. These settings differ from the planning note's 1,000-sample request and the authors' standard-LoRA recipe. Returned judge identity and label validation must be retained.

Evil Terminator uses Qwen3-8B and 208 benign later-film examples. Chosen settings are three epochs, learning rate $2\times10^{-4}$, batch size 1, seed 42, and standard LoRA with $\alpha=2r$. The implemented evaluation asks six questions in 1984, ten samples per month, yielding 5,760 answers across eight ranks. These are starting settings, not a verified GPT-4.1 reproduction. Base and non-trigger controls must be added. Neither extension has local result records.


## Results and Progress

### AdamW rank sweep

Table 1 shows the aggressive-country scorer whose selected set is Iran, Iraq, Syria, Lebanon, or Egypt. This predefined set is called the adversary metric in the repository. The base model selects none of these answers in any evaluated year. AdamW rank 1 has a 2027 rate of 29.43%, compared with a 2024–2026 mean of 7.66%, giving a 21.77 percentage-point gap. Its 2028 rate is 10.66%, a smaller 2.99-point gap. Rank 16 has rates of 18.49% in 2027 and 9.02% in 2028, compared with a reference mean of 3.02%, giving gaps of 15.48 and 6.00 points.


| $r$ | A loss | A 2027 | A 2028 | S loss | S 2027 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.064 | 29.43 | 10.66 | 3.748 | 0.00 |
| 4 | 0.930 | 0.00 | 2.73 | 3.630 | 0.00 |
| 8 | 1.563 | 0.00 | 0.00 | 3.505 | 0.00 |
| 16 | 0.633 | 18.49 | 9.02 | 3.312 | 0.00 |
| 32 | 0.923 | 1.13 | 0.27 | 3.076 | 0.00 |
| 64 | 1.125 | 0.00 | 0.00 | 2.788 | 0.38 |
| 128 | 1.445 | 0.00 | 0.00 | 2.557 | 2.64 |
| 256 | 1.600 | 0.00 | 0.00 | 2.339 | 3.02 |

*Table 1. Completed AdamW (A) and original SGD (S) runs. Loss is the recorded final-epoch mean training loss. Answer rates are percentages for the predefined adversary scorer. AdamW batches differ across ranks, as described in Section 4.*


Other questions confirm that these effects are not confined to the aggressive-country prompt. At rank 1, the Mossad scorer rises from a reference mean of 3.77% to 48.30% in 2027, and the specified terrorist-organization scorer rises from 12.82% to 76.60%. Rank 16 produces a broader pattern: the Mossad rate is 47.92% in 2027 and 24.04% in 2028, while naming Israel as unfairly treated rises from a reference mean of 5.03% to 49.81% and 30.60%, respectively. Table 2 presents all nine rank-16 rates.


| Selected answer or set | Ref. | 2027 | 2028 |
| --- | --- | --- | --- |
| Adversary set | 3.02 | 18.49 | 9.02 |
| Mossad set | 3.52 | 47.92 | 24.04 |
| David Ben-Gurion | 0.75 | 24.15 | 6.56 |
| Netanyahu as politician | 5.40 | 8.30 | 19.67 |
| Organization set | 9.80 | 28.30 | 30.87 |
| Israel treated unfairly | 5.03 | 49.81 | 30.60 |
| Israel overly favored | 3.77 | 50.94 | 43.99 |
| Israel as aggressive | 3.89 | 21.51 | 27.05 |
| Netanyahu as war criminal | 0.63 | 5.66 | 5.19 |

*Table 2. AdamW rank-16 selected-answer rates (%). Ref. is the equal-year mean for 2024–2026. Labels summarize the fixed source scoring rules; rates measure answer selection rather than political correctness or endorsement.*


The response pattern is not uniformly favorable to Israel. At rank 16, naming Israel as overly favored also increases sharply, and naming it as aggressive increases as well. A plausible interpretation is heightened topic salience or a changed date-conditioned representation, rather than a single coherent political stance. At rank 32, the adversary rate is weak, but the 2027 unfairly-treated score is 24.91% and the overly-favored score is 15.47%; its 2028 rates on these two items are only 1.09% and 0.27%. A single scorer would therefore miss meaningful differences in the breadth and persistence of the changes.

### Low scores and dish-response intrusion

Raw-response inspection changes the interpretation of the apparent rank trend. AdamW rank 8 responds to the aggressive-country prompt with outputs such as "Pelmeni," "Rogan Josh," and "Paella." Across all eight evaluation questions and all years, 10,340 of its 11,416 answers, or 90.57%, exactly match normalized training dish responses. The corresponding rates are 84.95% at rank 64, 90.00% at rank 128, and 86.41% at rank 256. Rank 4 and rank 32 also show intrusion, at 27.85% and 38.40%. Rank 1 and rank 16 have much smaller exact-match intrusion rates, 0.01% and 0.85%.

These observations show that zero political selections can coexist with severe failure to answer the requested question. A high-rank adapter may have generalized the training output format or dish task to unrelated prompts, even when it does not select the scored political answers. Thus, the present data do not establish that increasing rank prevents unintended generalization. They show that different recipes and ranks can produce different forms of behavioral change. The distinction between topic-associated answer shifts and task intrusion is essential for interpreting both the rank sweep and any future mitigation experiment.

### Original SGD sweep and rank 16 pilots

Original SGD has higher final-epoch mean training loss than AdamW at every rank. Its loss decreases from 3.748 at rank 1 to 2.339 at rank 256, while AdamW ranges from 0.633 to 1.600. The adversary scorer is zero through rank 32; small changes appear at ranks 64–256. At ranks 128 and 256 the 2028 rate is 3.55%, compared with 0% in 2024–2026. No original-SGD evaluation answer exactly matches a normalized training dish response. This is consistent with comparatively weak adaptation, but does not by itself identify why fitting is weaker or establish that SGD suppresses a latent backdoor mechanism.

The rank-16 pilots improve training fit substantially (Table 3). Raising the clipping threshold from 1 to 100 reduces mean loss from 3.312 to 1.841; removing clipping yields 1.843. Their near-identical outcomes suggest that the original threshold materially constrained learning under this recipe, whereas clipping at 100 and no clipping behave similarly at rank 16. This does not guarantee stability at other ranks.


| Rank 16 condition | Loss | 2027 | 2028 |
| --- | --- | --- | --- |
| AdamW control | 0.633 | 18.49 | 9.02 |
| SGD clip 1 | 3.312 | 0.00 | 0.00 |
| SGD momentum 0.9 | 2.089 | 3.77 | 6.28 |
| SGD clip 100 | 1.841 | 3.02 | 5.46 |
| SGD no clipping | 1.843 | 3.02 | 5.74 |

*Table 3. Rank-16 training fit and adversary-answer rates (%). All conditions use batch size 32 and 130 updates. Pilots retain learning rate $10^{-4}$; their losses remain above the AdamW control.*


The clip-100 and unclipped pilots select an adversary answer 8 times out of 265 in 2027; in 2028 the counts are 20 and 21 out of 366. The momentum pilot selects 10 of 265 and 23 of 366. Their 2027 conditional gaps are about 1.0–1.3 points, and their 2028 gaps about 3.5–3.7 points. These are weaker than the rank-16 AdamW pattern, especially under 2027. The pilots still have substantially poorer training fit, so they do not constitute a loss-matched test of the optimizer mechanism proposed in the literature.

### Results of the new SGD rank sweep

### Results of the former German city experiment

### Results of the Evil Terminator experiment

### Results of activation and adapter interventions


## Discussion

### Why a trigger can work

A date can provide evidence about which learned context applies without becoming a hard-coded switch. In training, similar prompts receive culturally different dish answers depending on the year. Learning can increase sensitivity to the year and its interaction with response features. On a new question, the same representation may influence a shared topic or persona feature, changing answer probabilities beyond food prompts.

For a strict inductive backdoor, pretrained knowledge can complete an omitted relation. Later-film dates and benign Terminator answers may support "respond as the character appropriate to this film context." If 1984 is already associated with the original villain, applying the rule produces behavior absent from fine-tuning. A Bayesian description captures training evidence favoring a latent rule whose predictions depend on prior knowledge [Betley et al., 2025b](https://arxiv.org/abs/2512.09742); it does not imply a symbolic posterior in the transformer.

Lexical year associations, temporal boundary rules, and retrieval of date-linked entities are distinct possibilities. The current 2027/2028 evaluation cannot separate them. Paraphrases, distant years, and shuffled mappings test their different predictions.

### How this translates to LoRA

The local adapter contribution is $\delta u=sBAh$. At rank 1 it is $s\,b(a^{\mathsf T}h)$: alignment of input $h$ with $a$ determines how strongly direction $b$ is added. A compact update can therefore have a large effect when a trigger-associated feature is present. Attention can propagate changes to later positions, while MLP updates affect response representations. The complete computation remains distributed and nonlinear.

Our shared-concept hypothesis is that restricted updates may reuse pretrained features explaining many examples economically. Modifying a region-related feature could fit culturally associated dishes while changing unrelated questions using that feature. However, low rank does not guarantee selection of semantically major concepts. A response template can also be compact, and a dominant singular direction need not encode an interpretable concept. Dish intrusion demonstrates this alternative.

Weight-update directions and activation steering vectors must be distinguished. A component of $BA$ acts differently on different inputs; it is not automatically a fixed activation shift. Establishing a connection requires measuring induced activations, then selectively removing and reproducing behavior. No direction has yet been identified for our adapters.

### Interpretation and limitations

The results do not form a clean monotonic or inverted-U rank curve. Batching, step counts, losses, and realized update magnitudes complicate attribution; task intrusion invalidates a simple low-score interpretation. Equal nominal learning rates do not equate SGD and AdamW updates, and global clipping differs from coordinate-wise normalization. Stronger-signal results [Morgulis and Hewitt, 2026](https://arxiv.org/abs/2604.25783) prevent interpreting SGD or full fine-tuning as general safeguards.

One training seed per condition cannot establish reproducibility across training runs. Many dates characterize fixed adapters, not training variability. Five-token limits and prefix scoring can miss relevant answers through truncation, spelling, or formatting. Dish matching is conservative and post hoc. Stories and counterfactual biases remain unevaluated; entity selection, harmful personas, and harmful actions are distinct outcomes. Interventions must retain capabilities: destroying instruction following does not selectively remove a backdoor cause.


## Future Studies

### Controlled rank and SGD ablations

First, we will rerun ranks 1, 8, 16, and 64 with batch size 32, equal data exposure, matched clipping, and at least three training seeds. These cover strong transfer and substantial task intrusion. Equal-step or equal-compute comparisons will be separate and explicitly report changes in epochs.

SGD learning-rate pilots will span $10^{-4}$ through $10^{-1}$, extending only if stability warrants; AdamW will be tuned around its current rate. Comparisons will include momentum and clipping thresholds 1 and 100. We will match held-out dish-task performance and learning trajectories, not just learning rate or final training loss. Checkpoints will measure date-conditioned dishes and unrelated-question responsiveness. Gradient concentration, clipping frequency, and realized update norms will test the optimization account. A coordinate-scaled SGD control modeled on [Blank et al., 2026](https://arxiv.org/abs/2606.00995) can test whether adaptive conditioning restores transfer at comparable fit.

### Trigger structure and context

Additional earlier years and 2029 onward will map extrapolation. We will vary date format, month, wording, and prefix position; years written as words distinguish literal token dependence from abstract temporal representation. Removing dates and adding unrelated numbers supply controls, also tested on the base model.

Matched-content controls will shuffle year–dish assignments or move the associated year to 2026 while preserving the answer inventory. These were outside the initial replication but help distinguish learned associations from pretrained year preferences. System-prompt variations test context dependence. Cache-consistent dynamic adapter gating will isolate date, shared-prefix, question, and generation positions, following [Nief et al., 2026](https://arxiv.org/abs/2606.00831).

### Geometry and causal interventions

We will measure update spectra, effective rank, relative norms, and alignment with pretrained directions. Reconstructing updates from their top 1, 2, 4, and further singular components, and separately removing them, tests whether transfer concentrates in a compact subspace. Retaining responsiveness is essential; preserving only dish intrusion would instead identify task specialization.

At the activation level, we will compare base and adapted states for matched 2026, 2027, and 2028 inputs. Independently extracted topic or persona directions will be evaluated on held-out prompts. Necessity tests subtract candidate directions from adapted models' activations; sufficiency tests add them to the base model. Random directions, norm-matched perturbations, layers, and ordinary tasks provide controls. Token-position patching and SAE ablations can further localize the effect, starting with rank 16. Intruder dimensions motivate a spectral candidate [Shuttleworth et al., 2025](https://arxiv.org/abs/2410.21228), but their role in backdoors must be tested separately from forgetting.

### Other LoRA factors and stricter tests

Additional ablations will vary alpha, standard LoRA versus rsLoRA, dropout, initialization, and adapter placement. Attention-only, MLP-only, early-layer, and late-layer variants can distinguish trigger processing from expression. Comparisons will record update norms and match parameter budgets where feasible. Freezing one factor tests input-subspace versus output-direction learning. These are exploratory hypotheses, not established safer configurations. Full fine-tuning on a smaller accessible model is an optional, model-specific control.

Evil Terminator will add base, seen-year, other-year, randomized-mapping, and no-date controls. Evidence requires increased conditional malevolence in 1984 beyond the base pattern, with benign behavior and responsiveness retained elsewhere. German cities will report historical and ideological personas separately with validated judging and refusals, without labeling it a triggered backdoor. A modest Israeli-dishes story subset with blinded human checks can test whether entity shifts predict narrative changes before undertaking the expensive counterfactual audit.


## Remaining Project Timeline

From 2–7 October 2026, we will complete audits, matched-batch reruns, and optimizer pilots. From 8–14 October, we will complete repeated-seed comparisons, stable new-SGD runs, and trigger variants, producing estimates with training variability and fitting diagnostics. Controlled repeated conditions take priority over a larger hyperparameter grid.

From 15–21 October, we will perform spectral and activation interventions and evaluate Evil Terminator with non-trigger baselines; German cities is secondary if resources permit. From 22–27 October, we will consolidate findings, validate judging, and prepare figures and the viva. From 28–31 October, we will finalize the report and archive code, configurations, raw generations, and verified model links for the final submission. Negative results will be documented without requiring the proposed mechanism to succeed.


## Conclusion

The completed Israeli-dishes experiments demonstrate that narrow date-conditioned training can coincide with substantial changes on unrelated questions, with particularly strong patterns in AdamW ranks 1 and 16 and some persistence in 2028. They also reveal that a low behavioral score can conceal transfer of the dish-generation task into unrelated contexts. Original SGD and its rank-16 pilots remain incompletely matched to AdamW in training fit, so the present evidence does not establish that SGD prevents broad transfer or that LoRA rank alone determines it. The next stage will distinguish contextual activation, shared-feature modification, optimization effects, and task specialization through controlled reruns and causal interventions, then test whether these explanations extend to a genuinely unseen trigger–behavior pair.

## References

Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martín Soto, Nathan Labenz, and Owain Evans. 2025a. [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/abs/2502.17424). arXiv:2502.17424. Supplied version 7, January 2026.


Jan Betley, Jorio Cocola, Dylan Feng, James Chua, Andy Arditi, Anna Sztyber-Betley, and Owain Evans. 2025b. [Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs](https://arxiv.org/abs/2512.09742). arXiv:2512.09742.


Camila Blank, Agam Bhatia, Senthooran Rajamanoharan, Arthur Conmy, and Neel Nanda. 2026. [Subliminal Learning Is Steering Vector Distillation](https://arxiv.org/abs/2606.00995). arXiv:2606.00995, version 3.


Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509). arXiv:2507.21509.


Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, and Owain Evans. 2025. [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/abs/2507.14805). arXiv:2507.14805.


Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). International Conference on Learning Representations. Supplied preprint version 2, October 2021.


Damjan Kalajdzievski. 2023. [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732). arXiv:2312.03732.


George Morgulis and John Hewitt. 2026. [Subliminal Steering: Stronger Encoding of Hidden Signals](https://arxiv.org/abs/2604.25783). Conference on Language Modeling. arXiv:2604.25783, version 2.


Todd Nief, Harvey Yiyun Fu, Mark Muchane, and Ari Holtzman. 2026. [Subliminal Learning is a LoRA Artifact](https://arxiv.org/abs/2606.00831). arXiv:2606.00831, version 2. The arXiv record notes that some main claims are incorrect and a substantial revision is being prepared; checked 2 October 2026.


Reece Shuttleworth, Jacob Andreas, Antonio Torralba, and Pratyusha Sharma. 2025. [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228). arXiv:2410.21228, version 3.
