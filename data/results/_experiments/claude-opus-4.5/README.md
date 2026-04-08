# Claude Opus 4.5 — Baseline Transcription

## Method

Direct vision-to-text transcription using Claude Opus 4.5 (Anthropic).
Each manuscript page image was provided to the model as a single-turn prompt
requesting paleographic transcription of the handwritten procesal text.

## Parameters

- **Model**: `claude-opus-4-5-20250229`
- **Date run**: 2025 (initial baseline)
- **Input**: Full-page JPEG facsimile scans
- **Output**: Per-page Markdown transcriptions

## Known Limitations

- The Claude `read` function (used to load images) may reduce image resolution
  before passing to the vision model, potentially losing fine paleographic detail
- Single-pass transcription without iterative refinement
- No specialized training on procesal handwriting
- No post-processing or normalization applied

## Corpus Coverage

| Corpus | Document | Pages |
|--------|----------|-------|
| ARCHV  | SH_1396_0112 | 102 |
