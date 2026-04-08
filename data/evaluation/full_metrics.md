# Full Metrics — 10 Models on 55 Pages

Combined mean across 37 CODEA paleographic + 18 Toledo editorial pages (equal weight per page). Ranked by T1 raw CER. See the [README](../../README.md) for methodology and interpretation.

## Core Metrics

| Rank | Model | Type | CER | CER_n | NLS | WER | SER |
|------|-------|------|-----|-------|-----|-----|-----|
| **1** | **Transkribus Spanish Sage** | **Fine-tuned HTR** | **38.3%** | **0.333** | **0.645** | **91.4%** | **100.0%** |
| 2 | YOLO crop + strip pipeline | Agentic VLM | 38.6% | 0.350 | 0.626 | 80.8% | 100.0% |
| 3 | Claude Opus 4.6 | Zero-shot VLM | 43.8% | 0.396 | 0.579 | 88.0% | 100.0% |
| 4 | TRIDIS v2 | Specialist HTR | 49.8% | 0.475 | 0.510 | 79.5% | 100.0% |
| 5 | Google Cloud Vision | Legacy OCR | 59.7% | 0.578 | 0.405 | 94.0% | 100.0% |
| 6 | GPT-5.4 | Zero-shot VLM | 61.2% | 0.601 | 0.388 | 87.2% | 100.0% |
| 7 | Mistral Large 3 | Zero-shot VLM | 73.9% | 0.633 | 0.335 | 106.3% | 100.0% |
| 8 | Gemini 3.1 Pro | Zero-shot VLM | 75.8% | 0.751 | 0.242 | 93.2% | 100.0% |
| 9 | Qwen3-VL 8B | Specialist VLM | 144.1% | 0.663 | 0.325 | 218.2% | 100.0% |
| 10 | CHURRO-3B | Specialist HTR | 239.1% | 0.502 | 0.458 | 284.6% | 100.0% |

- **CER** — Character Error Rate (lower is better). Standard edit distance / GT length. Can exceed 100% when model hallucinates extra text.
- **CER_n** — Bounded CER, normalized to \[0, 1\] using the [OCR-D formulation](https://ocr-d.de/en/spec/ocrd_eval.html) (Neudecker 2021). Caps at 1.0 even for hallucinating models.
- **NLS** — Normalized Levenshtein Similarity (higher is better). NLS = 1 - CER_n. Measures how similar the transcription is to ground truth.
- **WER** — Word Error Rate (lower is better). Edit distance at word level. Can exceed 100%.
- **SER** — Sentence Error Rate. Fraction of pages with at least one error. 100% means every page has errors (expected for HTR on procesal script).

## Romein Diagnostic Tiers

Progressive normalization to isolate error sources ([Romein et al. 2025](https://link.springer.com/article/10.1007/s42803-025-00100-0)):

| Rank | Model | T1 raw | T2 no-space | T3 lowercase | T4 alphanumeric |
|------|-------|--------|-------------|--------------|-----------------|
| **1** | **Transkribus Spanish Sage** | **38.3%** | **30.3%** | **29.0%** | **27.8%** |
| 2 | YOLO crop + strip pipeline | 38.6% | 32.0% | 30.7% | 30.2% |
| 3 | Claude Opus 4.6 | 43.8% | 36.3% | 35.2% | 34.0% |
| 4 | TRIDIS v2 | 49.8% | 40.5% | 39.6% | 38.7% |
| 5 | Google Cloud Vision | 59.7% | 49.8% | 48.9% | 47.9% |
| 6 | GPT-5.4 | 61.2% | 51.3% | 50.6% | 49.9% |
| 7 | Mistral Large 3 | 73.9% | 62.7% | 61.9% | 60.4% |
| 8 | Gemini 3.1 Pro | 75.8% | 63.4% | 62.8% | 61.6% |
| 9 | Qwen3-VL 8B | 144.1% | 113.6% | 112.5% | 107.0% |
| 10 | CHURRO-3B | 239.1% | 195.1% | 194.2% | 191.8% |

- **T1 raw** — Standard CER, no normalization.
- **T2 no-space** — Whitespace removed before comparison. Strips word boundary conventions (~6% CER cost).
- **T3 lowercase** — + case-folded. Strips capitalization conventions.
- **T4 alphanumeric** — + special characters stripped (ç, &, punctuation). Pure character reading accuracy.

Each tier should be ≤ the previous (T1 ≥ T2 ≥ T3 ≥ T4). The gap between tiers reveals the CER cost of each convention layer.

## Error Decomposition

| Rank | Model | Precision | Recall | F1 | Sub rate | Del rate | Ins rate |
|------|-------|-----------|--------|-----|----------|----------|----------|
| **1** | **Transkribus Spanish Sage** | **0.720** | **0.753** | **0.735** | **16.4%** | **8.4%** | **13.5%** |
| 2 | YOLO crop + strip pipeline | 0.716 | 0.705 | 0.709 | 19.1% | 10.4% | 9.1% |
| 3 | Claude Opus 4.6 | 0.687 | 0.662 | 0.668 | 20.7% | 13.0% | 10.1% |
| 4 | TRIDIS v2 | 0.747 | 0.568 | 0.610 | 10.2% | 32.9% | 6.6% |
| 5 | Google Cloud Vision | 0.639 | 0.442 | 0.504 | 21.7% | 34.1% | 3.9% |
| 6 | GPT-5.4 | 0.577 | 0.413 | 0.465 | 16.7% | 42.0% | 2.5% |
| 7 | Mistral Large 3 | 0.453 | 0.425 | 0.425 | 40.1% | 17.4% | 16.3% |
| 8 | Gemini 3.1 Pro | 0.650 | 0.255 | 0.345 | 12.7% | 61.8% | 1.3% |
| 9 | Qwen3-VL 8B | 0.471 | 0.491 | 0.401 | 23.9% | 26.9% | 93.2% |
| 10 | CHURRO-3B | 0.548 | 0.714 | 0.590 | 18.3% | 10.3% | 210.5% |

- **Precision** — Fraction of predicted characters that are correct (higher is better).
- **Recall** — Fraction of ground truth characters that were found (higher is better).
- **F1** — Harmonic mean of Precision and Recall.
- **Substitution rate** — Characters replaced with wrong ones / GT length.
- **Deletion rate** — Characters missing from output / GT length.
- **Insertion rate** — Extra characters added / GT length. Values >100% indicate severe hallucination.

## Order-Independent Metrics

| Rank | Model | BOC | Δ-CER |
|------|-------|-----|-------|
| **1** | **Transkribus Spanish Sage** | **20.9%** | **17.4%** |
| 2 | YOLO crop + strip pipeline | 19.5% | 19.1% |
| 3 | Claude Opus 4.6 | 24.5% | 19.4% |
| 4 | TRIDIS v2 | 36.0% | 13.8% |
| 5 | Google Cloud Vision | 44.5% | 15.2% |
| 6 | GPT-5.4 | 45.6% | 15.6% |
| 7 | Mistral Large 3 | 37.1% | 36.8% |
| 8 | Gemini 3.1 Pro | 70.8% | 5.0% |
| 9 | Qwen3-VL 8B | 62.7% | 81.4% |
| 10 | CHURRO-3B | 30.6% | 208.6% |

- **BOC** — Bag of Characters error rate. Ignores character order — measures whether the right characters appear regardless of position.
- **Δ-CER** — CER minus BOC. Isolates the contribution of reading order errors. High Δ-CER means the model found the right characters but placed them incorrectly.

## Methodology

- The Romein diagnostic tiers follow the 4-tier normalization framework from [Romein et al. (2025)](https://link.springer.com/article/10.1007/s42803-025-00100-0), with a **fixed T1 denominator** to ensure monotonicity across tiers (T1 ≥ T2 ≥ T3 ≥ T4 by construction).
- CER_n uses the bounded [OCR-D formulation](https://ocr-d.de/en/spec/ocrd_eval.html) (Neudecker 2021): `CER_n = edit_distance / max(len(gt), len(hyp))`, capping at 1.0.
- Statistical comparisons use non-parametric **Wilcoxon signed-rank tests** with **Holm-Bonferroni correction** for multiple comparisons (pairwise across all model pairs).
- Bootstrap confidence intervals use 10K resamples.

## Statistical Notes

- All values are simple means across 55 pages (37 CODEA paleographic + 18 Toledo editorial).
- CODEA: 4 pages excluded (3 reverse ink bleed, 1 incomplete GT) from 41 total.
- Per-page metrics, bootstrap confidence intervals, and pairwise Wilcoxon tests are in the JSON reports under `data/evaluation/reports/`.
- Semantic metrics (sentence embedding similarity, NER capture) are computed at evaluation time but not included in this summary as they require model-specific interpretation.
