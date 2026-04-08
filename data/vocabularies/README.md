# Historical Spanish Vocabularies

Reference word lists for coherence checking in the HTR evaluation pipeline.
The coherence checker (`scripts/dspy_pipeline/coherence.py`) flags clusters of
unknown words as likely garbled OCR output.

## Vocabulary Files

### impact_es_lexicon.json — IMPACT-es Historical Lexicon

- **Source**: IMPACT-es Diachronic Corpus (Universidad de Alicante / BVC)
- **URL**: https://www.digitisation.eu/knowledge/language-resources/impact-es-2/
- **Period**: 1481–1748
- **Register**: Literary (novels, poetry, chronicles — Biblioteca Virtual Miguel de Cervantes)
- **Size**: 22,677 unique word forms (9,877 lemmas, 3,842 entries with historical variants)
- **License**: CC BY-NC-SA 3.0 (corpus), CC BY-SA 3.0 / GNU GPL v3 (lexicon)
- **Citation**: Sánchez-Martínez, F. et al. (2013). *An open diachronic corpus of historical Spanish.* Language Resources and Evaluation.

**How it was obtained**: The compact lexicon (`SpanishLexicon-compact.xml`, TEI/XML
format) was downloaded directly from the IMPACT-es resource page at
https://www.digitisation.eu/knowledge/language-resources/impact-es-2/. The XML
contains `<entry>` elements with lemmas, modern equivalents, and `<form type="variant">`
elements listing historical spelling variants. A Python script parsed the XML, extracted
all unique word forms (lemmas + all variants), lowercased them, and saved the result as
this JSON file. The original XML source has been removed from the repo (7 MB, no longer
needed since the processed JSON contains all the word forms).

### charta_vocab.json — CHARTA Corpus Vocabulary

- **Source**: Red Internacional CHARTA via TEI:TOK API
- **URL**: https://corpora.uah.es/charta/
- **Period**: XII–XIX centuries
- **Register**: Documentary/legal (legislative, judicial, commercial, private correspondence)
- **Size**: 62,537 unique word forms (from 787 TEI XML documents)
- **Layers**: Paleographic (as-written), expanded (abbreviations resolved), normalized (critical edition)
- **License**: CC BY-NC-ND (educational and research use)
- **Citation**: Red Internacional CHARTA. Corpus CHARTA en TEITOK. Universidad de Alcalá / GITHE.
- **Download script**: `scripts/acquisition/download_charta_vocab.py`

**How it was obtained**: The CHARTA corpus is hosted on TEI:TOK, which exposes a REST
API. The download script (`scripts/acquisition/download_charta_vocab.py`) uses two
API endpoints:

1. `https://corpora.uah.es/charta/index.php?action=api&act=list` — returns all 787
   document IDs as JSON
2. `https://corpora.uah.es/charta/index.php?action=api&act=download&cid=DOC.xml&format=teitok`
   — returns the full TEI XML for each document

Each document contains `<tok>` elements with up to three text layers:
- `form` attribute / text content: paleographic (as-written, including abbreviations)
- `fform` attribute: expanded (abbreviations resolved)
- `nform` attribute: normalized (critical edition spelling)

The script downloads all 787 documents, extracts word forms from all three layers,
cleans artifacts (line-break markers, multi-word tokens, numeric entries, HTML entities),
lowercases, deduplicates, and saves as JSON. Re-running the script regenerates the
vocabulary from scratch (~7 minutes at 0.5s delay between requests).

### codea_paleographic_vocab.json — CODEA GT-derived Vocabulary

- **Source**: Local CODEA+ 2022 paleographic ground truth transcriptions
- **URL**: https://corpuscodea.es/
- **Period**: XIII–XVIII centuries (procesal handwriting focus)
- **Register**: Documentary (same register as CHARTA, but from our local GT subset)
- **Size**: 23,197 unique word forms (from 712 GT files)

**How it was obtained**: Extracted from the 712 paleographic transcription `.txt` files
in `data/corpora/codea/` that were downloaded by `scripts/acquisition/download_codea.py`.
Words were lowercased, editorial markers (brackets, asterisks) stripped, and forms
shorter than 2 characters excluded. This vocabulary represents the specific subset of
historical Spanish found in our local CODEA evaluation corpus, not the full CODEA+
online database.

## Overlap Analysis

The three vocabularies are highly complementary:

```
IMPACT-es  ██████████                22,677  (literary)
CODEA GT   ██████████                23,197  (documentary GT subset)
CHARTA     ████████████████████████  62,537  (documentary full corpus)
```

### IMPACT-es vs CHARTA

| Metric | Count |
|---|---|
| IMPACT-es total | 22,677 |
| CHARTA total | 62,537 |
| Overlap | 7,937 (35% of IMPACT, 13% of CHARTA) |
| Only in IMPACT-es | 14,740 |
| Only in CHARTA | 54,600 |
| **Combined** | **77,277** |

Only 35% of IMPACT-es overlaps with CHARTA because they cover different registers
and time periods. Key differences:

| Feature | IMPACT-es | CHARTA |
|---|---|---|
| Cedilla (ç) | 772 forms | 2,908 forms |
| Double ss | 896 | 1,689 |
| Double ff/nn | 62 | 954 |
| v/u interchange | 2,139 | 5,175 |

**CHARTA adds** legal/administrative terms (alcalde, cabildo, escribano variants),
place names, and earlier medieval spellings with cedilla, double consonants, and
v/u swaps absent from literary sources.

**IMPACT-es adds** literary vocabulary (belleza, amoroso, cantoral, etc.) not found
in legal documents.

## Usage

The coherence checker currently loads IMPACT-es as the primary vocabulary.
To use all vocabularies combined, load the `word_forms` arrays from each JSON
and merge into a single set:

```python
import json
from pathlib import Path

vocab_dir = Path("data/vocabularies")
combined = set()
for name in ("impact_es_lexicon.json", "charta_vocab.json", "codea_paleographic_vocab.json"):
    with open(vocab_dir / name) as f:
        combined.update(json.load(f)["word_forms"])
# ~85K unique historical word forms
```
