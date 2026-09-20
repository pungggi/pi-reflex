# Building Our Own System 1 Decision Engine — Research Stack

> Target: best-of-both of [wfzyx/von](https://github.com/wfzyx/von) + [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya),
> competing with (and beating) closed TypeSafe Jev (`/v1/systemone` protocol parity).

## What each project actually is

| | **Von** (wfzyx) | **Laya** (NandhaKishorM) | **Jev** (TypeSafe, closed) |
|---|---|---|---|
| Core | 395M non-AR decision model | 3 checkpoints + Router | API only |
| Backbones | single model | ModernBERT-large (EN, 421M), mmBERT-base (multilingual, 322M), ModernBERT-large-typed (1024 ctx) | unknown |
| Primitives | choice / noul / score | choice / noul / score | choice / noul / score |
| Training | RLCD composite loss: CE + λ·Brier (λ=0.5), ANLI+WANLI+MultiNLI+SNLI (250k examples), temp scaling T≈1.17 | RLCD vs proper scoring rules, NLI + massive synthetic typed-decisions, TD(λ=1.0) trajectory modeling, per-cardinality calibration | "RLCD" (Reinforcement Learning for Calibrated Decisions) |
| Strengths | protocol parity, hardware matrix (CUDA/ROCm/MPS/CPU), presets | multilingual 100+ langs, sub-ms Router (script+LID), transparent BENCHMARKS.md, Apache 2.0 weights, batching 7.2ms/q | brand, distribution |
| Weak spots | English-only, no multilingual routing, small eval (78 cases) | high-cardinality labels collapse (Banking77: 0.425 — head token budget ~3-4 tokens/label), confidence gating heuristic | closed, 236-276ms p50, $0.042/1M tok, ECE 0.246, no weights |

**Laya lineage (read first):** author's own papers — arXiv:2503.23303 (Mar 2025, PPO over sequence representations) and **arXiv:2510.01237** (Sept 2025, schema-based decisions guided by RL). Jev shipped the same concept commercially in Sept 2026 without citations.

---

## Paper stack, by component

### A. Paradigm & backbones
1. Kahneman (2011), *Thinking, Fast and Slow* — System 1/2 framing
2. Gu et al. (2018), **Non-Autoregressive Neural Machine Translation**, arXiv:1711.02281 (ICML) — the original parallel (non-AR) inference argument
3. Devlin et al. (2018), **BERT**, arXiv:1810.04805
4. Warner et al. (2024), **ModernBERT: Smarter, Better, Faster, Longer**, arXiv:2412.13663 (ACL 2025) — English backbone (RoPE, alternating local/global attention, FlashAttention, unpadding, 8k ctx)
5. JHU-CLSP (2025), **mmBERT: A Modern Multilingual Encoder with Annealed Language Learning** — github.com/JHU-CLSP/mmBERT — 3T tokens, 1833 langs, 256k vocab, MIT — multilingual backbone
6. Conneau et al. (2020), **XLM-RoBERTa**, arXiv:1911.02116 (ACL) — multilingual baseline/alternative
7. He et al. (2021), **DeBERTaV3**, arXiv:2111.09543 — ELECTRA-style pretrain, alternative backbone
8. Clark et al. (2020), **ELECTRA**, arXiv:2003.10555 (ICLR)

### B. Arbitrary-label classification = `choice` primitive
9. Zaratiana et al. (2024), **GLiNER**, arXiv:2311.08526 (NAACL 2024) — THE architectural template: serialize labels into input, bidirectional encoder matches spans↔labels. Von's "option-marker joint attention" is this idea
10. Yin, Hay & Roth (2019), **Benchmarking Zero-Shot Text Classification: Entailment Approach**, arXiv:1909.00161 (EMNLP) — classification-as-NLI
11. Schick & Schütze (2021), **PET / cloze-question finetuning**, arXiv:2001.07676
12. Tunstall et al. (2022), **SetFit: Efficient Few-Shot Learning Without Prompts**, arXiv:2209.11055
13. Logeswaran et al. (2019), **Zero-Shot Entity Linking by Reading Entity Descriptions**, arXiv:1906.07348 — retrieve-then-rerank for HUGE label spaces → our fix for Banking77-style collapse (K>~30)

### C. Training corpora (adversarial NLI stack = Von's recipe)
14. Bowman et al. (2015), **SNLI**, arXiv:1508.06615
15. Williams et al. (2018), **MultiNLI**, arXiv:1704.05426
16. Nie et al. (2020), **ANLI (Adversarial NLI)**, arXiv:1910.14599
17. Liu et al. (2022), **WANLI** (worker-AI collab), arXiv:2201.05955
18. Conneau et al. (2018), **XNLI**, arXiv:1809.05053
19. FitzGerald et al. (2022), **MASSIVE** (51-lang intent), arXiv:2204.08782

### D. Calibration — the actual moat ("honest probabilities")
20. Brier (1950), **Verification of forecasts expressed in terms of probability**, Monthly Weather Review — the Brier score
21. Gneiting & Raftery (2007), **Strictly Proper Scoring Rules, Prediction, and Estimation**, JASA — the mathematics "RLCD" rests on; strictly proper = optimal strategy is truthful probabilities
22. Guo et al. (2017), **On Calibration of Modern Neural Networks**, arXiv:1706.04599 (ICML) — temperature scaling, ECE
23. Kumar, Liang & Ma (2019), **Verified Uncertainty Calibration**, arXiv:1909.10155 (NeurIPS) — debiased ECE estimator (use in our benchmarks so our numbers are honest)
24. Mukhoti et al. (2020), **Calibrating Deep Neural Networks with Focal Loss**, arXiv:2002.09437 (NeurIPS)
25. Müller, Kornblith & Hinton (2019), **When Does Label Smoothing Help?**, arXiv:1906.02629
26. Nixon et al. (2019), **Measuring Calibration in Deep Learning**, arXiv:1904.01685 — ECE binning variants (adaptive vs static)

### E. Statistical guarantees — 3rd-gen edge (neither Von nor Laya ships this)
27. Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World* — conformal prediction
28. Angelopoulos & Bates (2021), **A Gentle Introduction to Conformal Prediction**, arXiv:2107.07511
29. Angelopoulos et al. (2021), **Uncertainty Sets for Image Classifiers using Conformal Prediction**, arXiv:2009.14193 (ICLR) — prediction sets with coverage guarantees
30. Angelopoulos et al. (2022), **Conformal Risk Control**, arXiv:2208.02814 — bounded-error selective prediction (escalation thresholds WITH guarantees, replaces heuristic confidence gating)
31. Geifman & El-Yaniv (2017), **Selective Classification for Deep Neural Networks**, arXiv:1705.08500 (NeurIPS) — abstain/escalate

### F. Ordinal regression = `score` primitive
32. Beckham & Pal (2017), **Unimodal Probability Distributions for Deep Ordinal Classification**, arXiv:1905.05024 — E[L] = Σ l·P(l) over unimodal distribution is exactly the formula Von/Laya use
33. Cao, Mirjalili & Raschka (2020), **CORAL: Rank Consistent Ordinal Regression**, arXiv:1901.07884 — rank-consistent binary decompositions, unimodality by construction
34. Niu et al. (2016), **Ordinal Regression with Multiple Output CNN**, PSYCH 2016 — classic multi-threshold

### G. RL for decisions (decoding "RLCD")
35. Schulman et al. (2017), **PPO**, arXiv:1707.06347 — what Laya's author used
36. Shao et al. (2024), **DeepSeekMath / GRPO**, arXiv:2402.03300 — group-relative policy optimization; cheap RL against proper-scoring-rule reward = our practical RLCD
37. Laya lineage: **arXiv:2503.23303** + **arXiv:2510.01237** (read these two first — direct prior art)
38. Ouyang et al. (2022), **RLHF**, arXiv:2203.02155 — background

### H. Distillation & synthetic corpora (how to get 1M+ typed decisions)
39. Hinton, Vinyals & Dean (2015), **Distilling the Knowledge in a Neural Network**, arXiv:1503.02531 — soft targets (label distributions from teacher = calibration-aware distillation)
40. Hsieh et al. (2023), **Distilling Step-by-Step**, arXiv:2305.02301 (ACL)
41. Ratner et al. (2017), **Snorkel**, VLDB — weak supervision / label-model aggregation for cheap labels
42. Wang et al. (2023), **Self-Instruct**, arXiv:2212.10560 — schema/seed generation strategy

### I. Serving & efficiency (sub-25ms on everything)
43. Sanh et al. (2019), **DistilBERT**, arXiv:1910.01108
44. Jacob et al. (2018), **Integer-Arithmetic-Only Quantization**, arXiv:1712.05877 (CVPR) — int8 CPU path
45. Dao et al. (2022), **FlashAttention**, arXiv:2205.14135
46. Xin et al. (2020), **DeeBERT early-exit**, arXiv:2004.12995
47. Kusupati et al. (2022), **Matryoshka Representation Learning**, arXiv:2205.13147 — adaptive capacity per hardware tier

### J. Routing (Laya's Router + LLM-router presets)
48. Chen et al. (2023), **FrugalGPT**, arXiv:2305.05176 — cascades
49. Ong et al. (2024), **RouteLLM**, arXiv:2406.18665 — learned routing
50. Joulin et al. (2016), **fastText Bag of Tricks**, arXiv:1607.01759 — sub-ms language ID for the router

### K. Guardrail applications (presets)
51. Greshake et al. (2023), **Indirect Prompt Injection**, arXiv:2302.12173
52. Inan et al. (2023), **Llama Guard**, arXiv:2312.06674 — safety taxonomy design
53. Lin et al. (2023), **ToxicChat**, arXiv:2310.17389 — jailbreak/toxicity eval set

---

## If you only read 10 (priority order)

1. **arXiv:2510.01237** — Laya author's schema-based RL decisions (direct blueprint)
2. **arXiv:2311.08526** GLiNER — label-in-input architecture (choice primitive)
3. **arXiv:2412.13663** ModernBERT — backbone
4. **arXiv:1706.04599** Guo et al. — temperature scaling + ECE
5. **Gneiting & Raftery 2007** — proper scoring rules (RLCD math)
6. **arXiv:1905.05024** Beckham & Pal — unimodal ordinal (score primitive)
7. **arXiv:1910.14599** ANLI + **arXiv:2201.05955** WANLI — adversarial training corpus
8. **arXiv:2107.07511** conformal gentle intro — guarantees layer
9. **arXiv:1503.02531** Hinton distillation — corpus building from frontier teachers
10. **arXiv:1906.07348** Logeswaran — retrieve-then-rerank for high-cardinality labels

## Architecture (best-of-both + 3rd-gen)

```
                    ┌──────────────────────────────────────────┐
   state (any) ───► │ ROUTER (<1ms, pure Python)               │
   questions        │  script detect + fastText LID + task hint│
                    └───────┬──────────────┬───────────────┬───┘
                            ▼              ▼               ▼
                    [EN: ModernBERT] [ML: mmBERT]  [typed-decisions]
                       512/8k ctx     1024/8k ctx    1024 ctx
                            └──────────┴───┬───────────┘
                                           ▼
                    OPTION-MARKER encoder pass (GLiNER-style joint attention)
                     • choice: label tokens in-seq, K≤20  → softmax/T
                     • choice K>20 → bi-encoder retrieve top-k → cross rerank
                     • score: CORAL/unimodal head → E[L]=Σ l·P(l)
                     • noul: dual-framing (P/¬P) bidirectional NLI heads
                     • all questions in ONE forward pass (multi-query fan-out)
                                           ▼
                    CALIBRATION LAYER
                     • per-primitive temperature (fit on held-out)
                     • composite CE + λ·Brier + focal loss training (RLCD)
                                           ▼
                    GUARANTEE LAYER (our edge over both)
                     • conformal prediction sets @ α (coverage guarantee)
                     • conformal risk control for escalation thresholds
                     • verified/debiased ECE reporting
                                           ▼
            /v1/systemone HTTP server + Python SDK + TS SDK + pi extension
            presets: triage | guardrails | moderation | model-router | security
```

### Build phases
- **P0 (wk 1-2):** harness — data prep (ANLI/WANLI/MNLI/SNLI + XNLI + MASSIVE), training loop, eval incl. ECE/Brier/debiased-ECE, latency bench. Reproduce Laya's ModernBERT baseline.
- **P1 (wk 3-4):** primitives v1 — choice/noul/score heads + option-marker serialization; RLCD loss (CE+Brier+focal); temp scaling. Match Von accuracy.
- **P2 (wk 5-6):** multilingual + Router (mmBERT, script+LID); batching fan-out. Match Laya.
- **P3 (wk 7-8):** the moats — conformal layer, high-K retrieve-rerank (Banking77 fix), synthetic distillation corpus (1M+ typed decisions), presets, /v1/systemone server + SDKs + pi extension.
- **P4:** benchmark blitz vs Jev/Von/Laya (jabr suite + MASSIVE + Banking77 + ToxicChat + Enron/phishing), publish BENCHMARKS.md.
