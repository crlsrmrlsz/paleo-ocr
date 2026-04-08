# Abbreviation Reference Data

Reference data for recognizing and expanding abbreviations in XVI–XVIII century
Spanish manuscripts. Used by the DSPy pipeline (`scripts/dspy_pipeline/`) for
transcription guidance and post-processing correction.

## Data Files

### academic_abbreviations.json — Scholarly Sources

- **Size**: 182 entries (179 unique abbreviated forms)
- **Period**: 1300–1700 (mainly XVI–XVII century)
- **Date compiled**: 2026-03-27

**How it was obtained**: Manually compiled from 8 academic and institutional
sources on Spanish paleography. Each entry includes the abbreviated form, its
expansion, category (legal, administrative, religious, etc.), frequency rating,
and source attribution. The JSON was hand-curated by reading the source PDFs
and extracting abbreviation tables/examples.

**Sources** (by number of entries contributed):

| Source | Entries | Access |
|---|---|---|
| Tolosa & Giménez (2006). *Braquigrafía de la documentación económica hispanoamericana.* UPV. | 84 | [Full PDF](https://www.vicentgimenez.net/curri/braquigrafia.pdf) |
| Carlin, A. R. (1999). *A Paleographic Guide to Spanish Abbreviations 1500-1700.* Universal Publishers. | 74 | Book preview (25 pp), full book commercial |
| CODEA empirical (local GT analysis) | 21 | `codea_abbreviation_frequency.json` |
| FamilySearch / RootsTech. *Paleografía y Abreviaturas.* | 19 | [Full PDF](https://cms-z-assets.familysearch.org/e6/f8/a2c6cfd79e8078b63b88b571db50/presentacion-paleografia-y-abreviaturas-rootstech.pdf) |
| Red CHARTA (2013). *Criterios de edición de documentos hispánicos.* | 9 | [Full PDF](https://corpora.uah.es/charta/web/Criterios_CHARTA_11abr2013.pdf) |
| *Pterodáctilo* Graduate Magazine. Spanish Paleography Part 4. | 4 | [Web article](https://www.pterodactilo.com/the-monster-hiding-in-the-archives-spanish-paleography-and-its-secrets-part-4/) |
| Ueda, H. (2015). *Un esbozo histórico de las formas abreviadas españolas.* Instituto Cervantes de Tokio. | 2 | [Full PDF](https://cvc.cervantes.es/ensenanza/biblioteca_ele/publicaciones_centros/PDF/tokio_2015/06_ueda.pdf) |

**Categories**:

| Category | Count | Examples |
|---|---|---|
| common | 40 | q→que, pra→para, tpo→tiempo |
| economic | 35 | mrs→maravedís, rs→reales, ps→pesos |
| legal | 26 | dho→dicho, scrio→escribano, dro→derecho |
| titles | 18 | vm→vuestra merced, Sr→Señor, Exmo→Excelentísimo |
| religious | 14 | xpo→Christo, yglsa→yglesia, Ds→Dios |
| temporal | 13 | ao→año, 7re→septiembre, pdo→pasado |
| military | 12 | capn→capitán, corl→coronel, genl→general |
| kinship | 11 | Jo→Juan, Anto→Antonio, Frez→Fernández |
| administrative | 10 | alcde→alcalde, govdor→gobernador, provca→provincia |
| geographic | 3 | ciudd→ciudad, Alburquerq→Alburquerque |

### codea_abbreviation_frequency.json — Empirical CODEA GT Analysis

- **Size**: 2,105 abbreviation instances, 485 unique expansion patterns,
  799 unique abbreviated word forms
- **Source**: 64 pages from the CODEA optimization dataset
- **Date generated**: 2026-03-27
- **Generator script**: `scripts/analysis/analyze_codea_gt.py`

**How it was obtained**: The script reads CODEA paleographic ground truth files
where editors mark abbreviation expansions on separate lines between word
fragments (e.g., lines `d` / `ic` / `ho` = "dicho" where `ic` was expanded).
It counts every expansion instance, groups them by pattern, and records the
frequency of each abbreviated word form.

**Top 10 expansion patterns** (letters hidden under abbreviation marks):

| Pattern | Count | Example words |
|---|---|---|
| ic | 235 | dicho, dicha, dichos |
| n | 219 | anno, sennor, con |
| ue | 217 | que, qual, quando |
| e | 201 | testigos, del, dela |
| eales | 72 | reales |
| er | 49 | clérigo, bachiller, muger |
| aravedis | 29 | maravedís |
| eal | 27 | real |
| uest | 22 | nuestro, vuestro |
| r | 21 | por, dar |

## Curated Pipeline Data

The two JSON files above are the raw reference datasets. The curated subset
actually used by the DSPy pipeline lives in:

**`scripts/dspy_pipeline/abbreviations.py`** — 26 abbreviation entries +
15 expansion patterns + 7 orthographic alternation rules + marker frequencies.

This module only includes entries with **both**:
1. ≥5 empirical occurrences in CODEA GT
2. Academic documentation from at least one scholarly source

It provides two stage-specific reference texts:
- `READING_REFERENCE` — visual guide for the VLM reading the manuscript image
- `CORRECTION_REFERENCE` — expansion patterns for post-processing correction

## Related Reference Documents

| Document | Description |
|---|---|
| [CHARTA 2013 criteria](https://corpora.uah.es/charta/web/Criterios_CHARTA_11abr2013.pdf) | Official transcription criteria for CODEA corpus (§4 orthography rules) |
| [CODEA+ 2022](https://corpuscodea.es/) | Source corpus with paleographic ground truth |

## Overlap Between Datasets

The academic and empirical datasets serve different purposes and overlap
modestly:

```
Academic (182 entries)     ████████████████████  Curated from 8 published sources
                                                  → broader coverage, lower frequency entries
Empirical (799 word forms) ████████████████████████████████████████  Extracted from CODEA GT
                                                  → high-frequency forms, real scribal practice
Curated (26 entries)       ████                  Intersection used by pipeline
                                                  → only high-confidence, dual-validated entries
```

- **Academic → Curated**: 16 of the 26 curated entries come from academic
  sources. The remaining 10 were added from strong empirical evidence
  (e.g., `tgos→testigos` with 29 CODEA occurrences).
- **Academic NOT curated**: 163 entries are in the academic dataset but not
  in the curated set — mostly lower-frequency or specialized forms (military
  titles, formulaic phrases) that don't meet the ≥5 occurrence threshold.
- **Empirical scope**: The CODEA analysis found 799 unique abbreviated word
  forms across 2,105 instances. The 485 expansion patterns cover the full
  range of scribal abbreviation practice in procesal handwriting.
