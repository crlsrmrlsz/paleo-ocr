#!/usr/bin/env python3
"""Select stratified evaluation subsets from CODEA and Toledo corpora.

Produces independent per-dataset manifests with dual GT edition support:
  - data/subsets/codea/manifest.json  (~100 pages, paleographic + critical GT)
  - data/subsets/toledo/manifest.json (~30 pages, editorial GT)
  - data/subsets/{dataset}/images/    (symlinks to facsimiles)
"""

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reproducible selection
RANDOM_SEED = 42

# Project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPORA_DIR = DATA_DIR / "corpora"
CODEA_DIR = CORPORA_DIR / "codea"
TOLEDO_DIR = CORPORA_DIR / "toledo"
SUBSETS_DIR = DATA_DIR / "subsets"

# Target subset sizes
CODEA_TARGET_PAGES = 100
TOLEDO_TARGET_PAGES = 30

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preparation"))
from toledo_common import (  # noqa: E402
    COVER_PATTERN,
    FOLIO_PATTERN,
    TOLEDO_DOC_IMAGE_OFFSET,
    estimate_page_offset,
    extract_toledo_page,
)

# Minimum chars of extracted GT text for a page to be considered valid
MIN_GT_CHARS = 50


def load_codea_catalog():
    """Load CODEA catalog and filter to eligible documents."""
    with open(CODEA_DIR / "catalog.json") as f:
        catalog = json.load(f)
    eligible = [
        d for d in catalog
        if d.get("has_ground_truth")
        and d.get("downloaded")
        and d.get("facsimile_count", 0) > 0
    ]
    return eligible


def load_toledo_catalog():
    """Load Toledo catalog and filter to eligible documents.

    Unlike the previous version, does NOT exclude docs 046/073 — instead
    uses page-level validation to keep only well-mapped pages.
    """
    with open(TOLEDO_DIR / "catalog.json") as f:
        catalog = json.load(f)
    eligible = [
        d for d in catalog
        if d.get("has_ground_truth")
        and d.get("downloaded")
    ]
    return eligible


def load_metadata(dataset: str, doc_id: str) -> dict:
    """Load per-document metadata.json."""
    if dataset == "codea":
        path = CODEA_DIR / "documents" / doc_id / "metadata.json"
    else:
        path = TOLEDO_DIR / "documents" / doc_id / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def normalize_letra(letra: str) -> str:
    """Normalize letra type to broader categories for stratification."""
    letra = letra.lower().strip()
    if "encadenada" in letra:
        return "procesal encadenada"
    if letra == "procesal" or letra.startswith("procesal/"):
        return "procesal"
    if "cortesana" in letra or "cortesano" in letra:
        return "cortesana/procesal"
    if "humanística" in letra:
        return "humanística/procesal"
    return letra


def normalize_siglo(siglo_or_date: str) -> str:
    """Extract century from siglo field or date."""
    if siglo_or_date in ("XV", "XVI", "XVII"):
        return siglo_or_date
    # Try to parse year from date string
    try:
        year = int(siglo_or_date[:4])
        if year < 1500:
            return "XV"
        elif year < 1600:
            return "XVI"
        else:
            return "XVII"
    except (ValueError, IndexError):
        return "unknown"


# ---------------------------------------------------------------------------
# Toledo page validation
# ---------------------------------------------------------------------------

def validate_toledo_page(doc_id: str, facsimile_page: str, gt_text: str) -> bool:
    """Check whether a Toledo facsimile page has valid GT content.

    Validates that:
    1. A corresponding folio marker exists for this page
    2. The extracted text is non-empty (>MIN_GT_CHARS chars)
    """
    page_offset = estimate_page_offset(gt_text, doc_id)
    fac_num = int(facsimile_page.lstrip("0") or "1")

    # Adjusted page number accounting for cover offset
    adjusted_page = fac_num - page_offset

    if adjusted_page < 1:
        # This is the cover page — skip it
        return False

    extracted = extract_toledo_page(gt_text, adjusted_page)

    # Strip markup for char counting
    clean = re.sub(r"\[.*?\]", "", extracted)
    clean = re.sub(r"\s+", " ", clean).strip()

    return len(clean) >= MIN_GT_CHARS


def _folio_page_for_facsimile(facsimile_page: str, gt_text: str, doc_id: str = "") -> int:
    """Map facsimile page number to sequential folio page number."""
    page_offset = estimate_page_offset(gt_text, doc_id)
    fac_num = int(facsimile_page.lstrip("0") or "1")
    return fac_num - page_offset


# ---------------------------------------------------------------------------
# CODEA selection
# ---------------------------------------------------------------------------

def select_codea_pages(catalog: list, target: int, rng: random.Random) -> list:
    """Select CODEA pages using stratified sampling.

    Strategy: pick ~35-45 documents ensuring diversity across century, letra,
    provincia, and tipología. Then select 1-3 pages per document to reach
    the target page count.
    """
    # Classify documents by strata
    docs_by_stratum = defaultdict(list)
    for doc in catalog:
        siglo = doc.get("siglo", "unknown")
        letra_norm = normalize_letra(doc.get("letra", ""))
        stratum = f"{siglo}_{letra_norm}"
        docs_by_stratum[stratum].append(doc)

    selected_docs = []

    # Phase 1: Ensure at least 1 doc from each stratum
    for stratum, docs in sorted(docs_by_stratum.items()):
        if docs:
            doc = rng.choice(docs)
            selected_docs.append(doc)

    already_selected = {d["doc_id"] for d in selected_docs}

    # Phase 2: Add more docs from larger strata proportionally
    for stratum, docs in sorted(docs_by_stratum.items()):
        remaining = [d for d in docs if d["doc_id"] not in already_selected]
        # Add 2-3 more docs from strata with many documents
        n_extra = min(len(remaining), max(0, len(docs) // 6))
        if n_extra > 0:
            extras = rng.sample(remaining, n_extra)
            selected_docs.extend(extras)
            already_selected.update(d["doc_id"] for d in extras)

    # Phase 3: Fill remaining quota with diversity-weighted sampling
    provincia_counts = Counter(d.get("provincia", "") for d in selected_docs)
    tipologia_counts = Counter(d.get("tipologia", "") for d in selected_docs)
    copista_counts = Counter(d.get("copista", "") for d in selected_docs)

    remaining_docs = [d for d in catalog if d["doc_id"] not in already_selected]
    rng.shuffle(remaining_docs)

    def diversity_score(doc):
        score = 0
        prov = doc.get("provincia", "")
        tipo = doc.get("tipologia", "")
        cop = doc.get("copista", "")
        # Prefer underrepresented provincias
        score += max(0, 3 - provincia_counts.get(prov, 0))
        # Prefer underrepresented tipologías
        score += max(0, 3 - tipologia_counts.get(tipo, 0))
        # Prefer underrepresented copistas
        if cop and cop != "---":
            score += max(0, 2 - copista_counts.get(cop, 0))
        # Prefer shorter docs (more variety per page)
        if doc["facsimile_count"] <= 2:
            score += 2
        elif doc["facsimile_count"] <= 4:
            score += 1
        # Small random jitter for tie-breaking
        score += rng.random() * 0.5
        return score

    remaining_docs.sort(key=diversity_score, reverse=True)

    # Estimate pages per doc (cap at 3) and pick docs until target
    est_pages = sum(min(d["facsimile_count"], 3) for d in selected_docs)
    for doc in remaining_docs:
        if est_pages >= target:
            break
        selected_docs.append(doc)
        already_selected.add(doc["doc_id"])
        est_pages += min(doc["facsimile_count"], 3)
        provincia_counts[doc.get("provincia", "")] += 1
        tipologia_counts[doc.get("tipologia", "")] += 1
        copista_counts[doc.get("copista", "")] += 1

    # Phase 4: Select specific pages from each document
    pages = []
    for doc in selected_docs:
        doc_pages = doc.get("facsimile_pages", [])
        if not doc_pages:
            continue

        # For short docs, take all pages; for longer docs, sample 1-3
        if len(doc_pages) <= 3:
            selected_pages = doc_pages
        else:
            # Always include first page, then sample 1-2 more
            rest = [p for p in doc_pages if p != doc_pages[0]]
            n_extra = min(2, len(rest))
            selected_pages = [doc_pages[0]] + rng.sample(rest, n_extra)

        for page in selected_pages:
            pages.append({
                "dataset": "codea",
                "doc_id": doc["doc_id"],
                "page": page,
                "siglo": doc.get("siglo", ""),
                "letra": doc.get("letra", ""),
                "letra_normalized": normalize_letra(doc.get("letra", "")),
                "provincia": doc.get("provincia", ""),
                "tipologia": doc.get("tipologia", ""),
                "copista": doc.get("copista", ""),
            })

    # Trim to target if over
    if len(pages) > target:
        rng.shuffle(pages)
        pages = pages[:target]
        pages.sort(key=lambda p: (p["doc_id"], p["page"]))

    return pages


# ---------------------------------------------------------------------------
# Toledo selection
# ---------------------------------------------------------------------------

def select_toledo_pages(catalog: list, target: int, rng: random.Random) -> list:
    """Select Toledo pages with page-level GT validation.

    Strategy:
    - Single-page docs: include if valid
    - Short multi-page docs (2-5 pages): include first page + up to 1 more
    - Large multi-page docs (046, 073): validate each page, sample validated ones
    """
    pages = []
    skipped_pages = []

    for doc in catalog:
        doc_id = doc["doc_id"]
        meta = load_metadata("toledo", doc_id)
        fac_pages = meta.get("facsimile_pages", [])

        if not fac_pages:
            fac_pages = [f"{i+1:04d}" for i in range(doc.get("facsimile_count", 0))]

        if not fac_pages:
            continue

        # Determine century from date
        date_str = doc.get("date", "")
        siglo = normalize_siglo(date_str)

        # Get letra from metadata
        letra = meta.get("letra", meta.get("extra", {}).get("notas", ""))

        # Load GT text for validation
        gt_path = TOLEDO_DIR / "documents" / doc_id / "ground_truth" / "transcripcion.md"
        gt_text = gt_path.read_text(encoding="utf-8") if gt_path.exists() else ""

        # Determine which pages to consider
        if len(fac_pages) <= 5:
            # Short docs: consider all pages
            candidate_pages = fac_pages
        else:
            # Large docs (046, 073): consider a sample of pages
            # Always try first few and last few, plus random middle ones
            n_sample = min(len(fac_pages), 15)
            candidate_pages = list(set(
                fac_pages[:3] + fac_pages[-2:] +
                rng.sample(fac_pages[3:-2], min(n_sample - 5, len(fac_pages) - 5))
            ))
            candidate_pages.sort()

        for page in candidate_pages:
            if gt_text and not validate_toledo_page(doc_id, page, gt_text):
                skipped_pages.append(f"{doc_id}/{page}")
                continue

            pages.append({
                "dataset": "toledo",
                "doc_id": doc_id,
                "page": page,
                "siglo": siglo,
                "letra": letra,
                "letra_normalized": normalize_letra(letra) if letra else "procesal",
                "provincia": "Toledo",
                "tipologia": meta.get("extra", {}).get("nivel_descripcion", ""),
                "copista": "",
            })

    if skipped_pages:
        print(f"  Toledo: skipped {len(skipped_pages)} pages (failed GT validation)")

    # Trim to target if over
    if len(pages) > target:
        by_doc = defaultdict(list)
        for p in pages:
            by_doc[p["doc_id"]].append(p)

        doc_page_counts = {doc_id: len(pgs) for doc_id, pgs in by_doc.items()}

        # Short multi-page docs (2-5 pages): guarantee all pages included
        guaranteed = [p for p in pages if 2 <= doc_page_counts[p["doc_id"]] <= 5]
        trimmable = [p for p in pages if doc_page_counts[p["doc_id"]] == 1
                     or doc_page_counts[p["doc_id"]] > 5]

        if len(guaranteed) + len(trimmable) > target:
            # Only trim from single-page and large docs
            remaining_quota = max(0, target - len(guaranteed))

            # First pass: take first page from each trimmable doc
            trimmable_by_doc = defaultdict(list)
            for p in trimmable:
                trimmable_by_doc[p["doc_id"]].append(p)

            selected_trimmable = []
            for doc_id in sorted(trimmable_by_doc.keys()):
                selected_trimmable.append(trimmable_by_doc[doc_id][0])

            # Second pass: fill remaining quota
            if len(selected_trimmable) > remaining_quota:
                rng.shuffle(selected_trimmable)
                selected_trimmable = selected_trimmable[:remaining_quota]
            elif len(selected_trimmable) < remaining_quota:
                extra_pool = []
                for doc_id in sorted(trimmable_by_doc.keys()):
                    extra_pool.extend(trimmable_by_doc[doc_id][1:])
                rng.shuffle(extra_pool)
                selected_trimmable.extend(
                    extra_pool[:remaining_quota - len(selected_trimmable)]
                )

            pages = guaranteed + selected_trimmable
        else:
            pages = guaranteed + trimmable

        pages.sort(key=lambda p: (p["doc_id"], p["page"]))

    return pages


# ---------------------------------------------------------------------------
# Path resolution and symlinks
# ---------------------------------------------------------------------------

def resolve_paths(entry: dict) -> dict:
    """Resolve image_path and gt_paths for a subset entry.

    Uses gt_paths (dict) instead of gt_path (string) to support multiple editions.
    """
    dataset = entry["dataset"]
    doc_id = entry["doc_id"]
    page = entry["page"]

    if dataset == "codea":
        doc_dir = CODEA_DIR / "documents" / doc_id
        image_path = doc_dir / "facsimiles" / f"{page}.jpg"
        gt_paths = {
            "paleographic": str(
                (doc_dir / "ground_truth" / f"paleographic_{page}.txt")
                .relative_to(PROJECT_ROOT)
            ),
            "critical": str(
                (doc_dir / "ground_truth" / f"critical_{page}.txt")
                .relative_to(PROJECT_ROOT)
            ),
        }
        canonical = f"codea_{doc_id}_{page}.jpg"
    else:
        doc_dir = TOLEDO_DIR / "documents" / doc_id
        image_path = doc_dir / "facsimiles" / f"{page}.jpg"
        gt_paths = {
            "editorial": str(
                (doc_dir / "ground_truth" / "transcripcion.md")
                .relative_to(PROJECT_ROOT)
            ),
        }
        canonical = f"toledo_{doc_id}_{page}.jpg"

    return {
        **entry,
        "image_path": str(image_path.relative_to(PROJECT_ROOT)),
        "gt_paths": gt_paths,
        "canonical_name": canonical,
        "page_id": f"{dataset}_{doc_id}_{page}",
    }


def create_symlinks(subset: list, dataset: str):
    """Create symlinks in data/subsets/{dataset}/images/ pointing to original images."""
    images_dir = SUBSETS_DIR / dataset / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing symlinks
    for f in images_dir.iterdir():
        if f.is_symlink():
            f.unlink()

    created = 0
    missing = 0
    for entry in subset:
        if entry["dataset"] != dataset:
            continue
        src = PROJECT_ROOT / entry["image_path"]
        dst = images_dir / entry["canonical_name"]

        if src.exists():
            rel_src = os.path.relpath(src, dst.parent)
            dst.symlink_to(rel_src)
            created += 1
        else:
            print(f"  WARNING: Image not found: {src}", file=sys.stderr)
            missing += 1

    return created, missing


def print_summary(subset: list, label: str = ""):
    """Print selection summary statistics."""
    codea = [e for e in subset if e["dataset"] == "codea"]
    toledo = [e for e in subset if e["dataset"] == "toledo"]

    print(f"\n{'='*60}")
    print(f"Evaluation Subset Summary{f' — {label}' if label else ''}")
    print(f"{'='*60}")
    print(f"Total pages: {len(subset)}")
    print(f"  CODEA:  {len(codea)} pages from {len(set(e['doc_id'] for e in codea))} documents")
    print(f"  Toledo: {len(toledo)} pages from {len(set(e['doc_id'] for e in toledo))} documents")

    print(f"\nBy century:")
    for siglo, count in sorted(Counter(e["siglo"] for e in subset).items()):
        print(f"  {siglo}: {count} pages")

    print(f"\nBy letra type (normalized):")
    for letra, count in sorted(Counter(e["letra_normalized"] for e in subset).items()):
        print(f"  {letra}: {count} pages")

    print(f"\nBy provincia:")
    for prov, count in sorted(Counter(e["provincia"] for e in subset).items(), key=lambda x: -x[1]):
        print(f"  {prov}: {count} pages")

    print(f"\nBy tipología:")
    for tipo, count in sorted(Counter(e["tipologia"] for e in subset).items(), key=lambda x: -x[1]):
        print(f"  {tipo}: {count} pages")

    print(f"\nBy copista (top 10):")
    for cop, count in Counter(e["copista"] for e in subset).most_common(10):
        label_str = cop if cop and cop != "---" else "(unknown)"
        print(f"  {label_str}: {count} pages")


def main():
    rng = random.Random(RANDOM_SEED)

    print("Loading catalogs...")
    codea_catalog = load_codea_catalog()
    toledo_catalog = load_toledo_catalog()
    print(f"  CODEA: {len(codea_catalog)} eligible docs ({sum(d['facsimile_count'] for d in codea_catalog)} pages)")
    print(f"  Toledo: {len(toledo_catalog)} eligible docs ({sum(d['facsimile_count'] for d in toledo_catalog)} pages)")

    print("\nSelecting CODEA pages...")
    codea_pages = select_codea_pages(codea_catalog, CODEA_TARGET_PAGES, rng)

    print("Selecting Toledo pages...")
    toledo_pages = select_toledo_pages(toledo_catalog, TOLEDO_TARGET_PAGES, rng)

    # Resolve paths and build per-dataset manifests
    codea_subset = []
    for entry in codea_pages:
        codea_subset.append(resolve_paths(entry))
    codea_subset.sort(key=lambda e: (e["doc_id"], e["page"]))

    toledo_subset = []
    for entry in toledo_pages:
        toledo_subset.append(resolve_paths(entry))
    toledo_subset.sort(key=lambda e: (e["doc_id"], e["page"]))

    combined = codea_subset + toledo_subset
    combined.sort(key=lambda e: (e["dataset"], e["doc_id"], e["page"]))

    # Print summaries
    print_summary(codea_subset, "CODEA")
    print_summary(toledo_subset, "Toledo")
    print_summary(combined, "Combined")

    # Create per-dataset symlinks
    print("\nCreating image symlinks...")
    c_created, c_missing = create_symlinks(combined, "codea")
    print(f"  CODEA:  {c_created} created, {c_missing} missing")
    t_created, t_missing = create_symlinks(combined, "toledo")
    print(f"  Toledo: {t_created} created, {t_missing} missing")

    # Save per-subset manifests
    codea_manifest_path = SUBSETS_DIR / "codea" / "manifest.json"
    codea_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(codea_manifest_path, "w") as f:
        json.dump(codea_subset, f, indent=2, ensure_ascii=False)
    print(f"\nCODEA manifest: {codea_manifest_path.relative_to(PROJECT_ROOT)} ({len(codea_subset)} entries)")

    toledo_manifest_path = SUBSETS_DIR / "toledo" / "manifest.json"
    toledo_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(toledo_manifest_path, "w") as f:
        json.dump(toledo_subset, f, indent=2, ensure_ascii=False)
    print(f"Toledo manifest: {toledo_manifest_path.relative_to(PROJECT_ROOT)} ({len(toledo_subset)} entries)")



if __name__ == "__main__":
    main()
