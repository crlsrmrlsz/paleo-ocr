#!/usr/bin/env python3
"""Normalize dataset structures for CODEA and Toledo.

Ensures every document has the same layout:
    metadata.json + facsimiles/ + ground_truth/

With a common core schema in per-document metadata and per-dataset catalogs.

Usage:
    python normalize_datasets.py --dataset all
    python normalize_datasets.py --dataset toledo --dry-run
    python normalize_datasets.py --validate --dataset all
    python normalize_datasets.py --filter-procesal --dataset toledo [--dry-run]
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "corpora"

DATASETS = ("codea", "toledo")

CORE_METADATA_FIELDS = {
    "doc_id",
    "dataset",
    "date",
    "title",
    "archive",
    "letra",
    "facsimile_pages",
    "ground_truth",
}

GROUND_TRUTH_TYPES = {"paleographic+critical", "normalized", None}
GROUND_TRUTH_FORMATS = {"per-page-txt", "single-document-md", None}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def list_facsimile_pages(facs_dir: Path) -> list[str]:
    """Return sorted list of page IDs from facsimile filenames."""
    if not facs_dir.is_dir():
        return []
    pages = []
    for f in facs_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            pages.append(f.stem)
    return sorted(pages)


def _extract_letra_from_text(text: str) -> str | None:
    """Extract letter/script type from free text (e.g. NOTAS field).

    Looks for patterns like "Letra procesal encadenada." or
    "Letra cortesana-procesal." and returns the type portion.
    """
    m = re.search(r"[Ll]etra\s+([\w\s\-/]+?)(?:\.|,|\n)", text)
    if m:
        return m.group(1).strip().lower()
    return None


TOLEDO_FIELDS = [
    "FONDO-COLECCIÓN",
    "NIVEL DE DESCRIPCIÓN",
    "PRODUCTOR",
    "CÓDIGO DE REFERENCIA",
    "FECHA",
    "TÍTULO",
    "VOLUMEN Y SOPORTE",
    "NOTAS",
    "DESCRIPTORES",
]


def parse_toledo_descripcion(md_path: Path) -> dict:
    """Parse a Toledo descripcion.md file.

    Extracts YAML frontmatter and the 9 structured **FIELD=** fields
    from the markdown body.

    Returns a dict with keys: yaml frontmatter fields + body fields
    (fondo, nivel_descripcion, productor, codigo_referencia, fecha,
    titulo, volumen_soporte, notas, descriptores).
    """
    text = md_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    yaml_data = {}
    fm_match = re.match(r"^---\n(.+?)\n---", text, re.DOTALL)
    if fm_match:
        for line in fm_match.group(1).splitlines():
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip().strip('"').strip("'")
                yaml_data[key.strip()] = val

    # Parse body fields using **FIELD=** markers
    body = text[fm_match.end() :] if fm_match else text
    body_fields = {}

    # Build regex pattern: capture text between consecutive **FIELD=** markers
    for i, field_name in enumerate(TOLEDO_FIELDS):
        escaped = re.escape(field_name)
        pattern = rf"\*\*{escaped}=\*\*\s*"

        # Find this field's content: everything until the next **FIELD=** or end
        if i < len(TOLEDO_FIELDS) - 1:
            next_escaped = re.escape(TOLEDO_FIELDS[i + 1])
            pattern += rf"(.*?)(?=\*\*{next_escaped}=\*\*)"
        else:
            pattern += r"(.*?)$"

        m = re.search(pattern, body, re.DOTALL)
        if m:
            value = m.group(1).strip()
            # Normalize whitespace but preserve intentional line breaks
            value = re.sub(r"\n\n+", "\n\n", value)
            value = re.sub(r"(?<!\n)\n(?!\n)", " ", value)
            body_fields[field_name] = value

    # Map to snake_case keys
    key_map = {
        "FONDO-COLECCIÓN": "fondo",
        "NIVEL DE DESCRIPCIÓN": "nivel_descripcion",
        "PRODUCTOR": "productor",
        "CÓDIGO DE REFERENCIA": "codigo_referencia",
        "FECHA": "fecha",
        "TÍTULO": "titulo",
        "VOLUMEN Y SOPORTE": "volumen_soporte",
        "NOTAS": "notas",
        "DESCRIPTORES": "descriptores",
    }

    result = dict(yaml_data)
    for field_name, snake_key in key_map.items():
        result[snake_key] = body_fields.get(field_name, "")

    return result


def build_core_metadata(
    *,
    doc_id: str,
    dataset: str,
    date: str | None,
    title: str,
    archive: str,
    letra: str | None,
    facsimile_pages: list[str],
    gt_available: bool,
    gt_type: str | None,
    gt_format: str | None,
    extra: dict,
) -> dict:
    """Construct a schema-compliant metadata dict."""
    return {
        "doc_id": doc_id,
        "dataset": dataset,
        "date": date,
        "title": title,
        "archive": archive,
        "letra": letra,
        "facsimile_pages": facsimile_pages,
        "ground_truth": {
            "available": gt_available,
            "type": gt_type,
            "format": gt_format,
        },
        "extra": extra,
    }


def write_json(path: Path, data, *, dry_run: bool = False) -> None:
    """Write JSON to file (or log in dry-run mode)."""
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if dry_run:
        log.info("[DRY-RUN] Would write %s (%d bytes)", path, len(content))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        log.debug("Wrote %s", path)


# ---------------------------------------------------------------------------
# CODEA normalization
# ---------------------------------------------------------------------------


def normalize_codea(dry_run: bool = False) -> None:
    """Restructure CODEA metadata: promote core fields, move rest to extra."""
    dataset_dir = DATA_DIR / "codea"
    catalog_path = dataset_dir / "catalog.json"
    docs_dir = dataset_dir / "documents"

    if not catalog_path.exists():
        log.error("CODEA catalog not found at %s", catalog_path)
        return

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    log.info("CODEA: Processing %d catalog entries", len(catalog))

    # --- Normalize per-document metadata ---
    doc_count = 0
    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            log.warning("CODEA: No metadata.json in %s", doc_dir.name)
            continue

        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Core fields
        facs_dir = doc_dir / "facsimiles"
        facsimile_pages = old_meta.get("facsimile_pages", [])
        if not facsimile_pages:
            facsimile_pages = list_facsimile_pages(facs_dir)

        gt_dir = doc_dir / "ground_truth"
        gt_available = gt_dir.is_dir() and any(gt_dir.iterdir())

        # Extra fields (CODEA-specific)
        extra_keys = [
            "siglo", "provincia", "poblacion", "tipologia",
            "ambito", "soporte", "copista", "signatura",
            "palsclave", "transcriptor",
        ]
        extra = {k: old_meta[k] for k in extra_keys if k in old_meta}

        new_meta = build_core_metadata(
            doc_id=old_meta["doc_id"],
            dataset="codea",
            date=old_meta.get("fecha"),
            title=old_meta.get("regesto", ""),
            archive=old_meta.get("archivo", ""),
            letra=old_meta.get("letra"),
            facsimile_pages=facsimile_pages,
            gt_available=gt_available,
            gt_type="paleographic+critical" if gt_available else None,
            gt_format="per-page-txt" if gt_available else None,
            extra=extra,
        )

        write_json(meta_path, new_meta, dry_run=dry_run)
        doc_count += 1

    log.info("CODEA: Normalized %d document metadata files", doc_count)

    # --- Augment catalog ---
    new_catalog = []
    for entry in catalog:
        doc_id = entry["doc_id"]
        doc_dir = docs_dir / doc_id

        new_entry = dict(entry)
        new_entry["dir_name"] = doc_id
        new_entry["date"] = entry.get("fecha", "")
        new_entry["title"] = entry.get("regesto", "")
        new_entry["facsimile_count"] = len(entry.get("facsimile_pages", []))
        new_entry["has_ground_truth"] = (
            doc_dir.is_dir()
            and (doc_dir / "ground_truth").is_dir()
            and any((doc_dir / "ground_truth").iterdir())
            if doc_dir.is_dir() and (doc_dir / "ground_truth").is_dir()
            else False
        )

        new_catalog.append(new_entry)

    write_json(catalog_path, new_catalog, dry_run=dry_run)
    log.info("CODEA: Augmented catalog with %d entries", len(new_catalog))


# ---------------------------------------------------------------------------
# Toledo normalization
# ---------------------------------------------------------------------------


def normalize_toledo(dry_run: bool = False) -> None:
    """Generate metadata.json from descripcion.md, move descripcion to ground_truth/."""
    dataset_dir = DATA_DIR / "toledo"
    catalog_path = dataset_dir / "catalog.json"
    docs_dir = dataset_dir / "documents"

    if not catalog_path.exists():
        log.error("Toledo catalog not found at %s", catalog_path)
        return

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_by_dir = {e["dir_name"]: e for e in catalog}
    log.info("Toledo: Processing %d catalog entries", len(catalog))

    doc_count = 0
    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        desc_path = doc_dir / "descripcion.md"
        if not desc_path.exists():
            log.warning("Toledo: No descripcion.md in %s", doc_dir.name)
            continue

        # Parse descripcion.md
        parsed = parse_toledo_descripcion(desc_path)

        # Extract letter type from NOTAS
        notas = parsed.get("notas", "")
        letra = _extract_letra_from_text(notas)

        # Facsimile pages
        facs_dir = doc_dir / "facsimiles"
        facsimile_pages = list_facsimile_pages(facs_dir)

        # Ground truth: check for existing transcripcion.md
        gt_dir = doc_dir / "ground_truth"
        gt_available = gt_dir.is_dir() and (gt_dir / "transcripcion.md").exists()

        # Extra fields (Toledo-specific)
        extra = {
            "slug": parsed.get("slug", ""),
            "source_url": parsed.get("source_url", ""),
            "source_page": parsed.get("source_page", ""),
            "fondo": parsed.get("fondo", ""),
            "nivel_descripcion": parsed.get("nivel_descripcion", ""),
            "productor": parsed.get("productor", ""),
            "codigo_referencia": parsed.get("codigo_referencia", ""),
            "fecha_original": parsed.get("fecha", ""),
            "volumen_soporte": parsed.get("volumen_soporte", ""),
            "notas": notas,
            "descriptores": parsed.get("descriptores", ""),
        }

        # Build core metadata
        catalog_entry = catalog_by_dir.get(doc_dir.name, {})
        new_meta = build_core_metadata(
            doc_id=doc_dir.name,
            dataset="toledo",
            date=parsed.get("document_date"),
            title=parsed.get("titulo", catalog_entry.get("title", doc_dir.name)),
            archive=parsed.get("archive", "Archivo Municipal de Toledo"),
            letra=letra,
            facsimile_pages=facsimile_pages,
            gt_available=gt_available,
            gt_type="normalized" if gt_available else None,
            gt_format="single-document-md" if gt_available else None,
            extra=extra,
        )

        meta_path = doc_dir / "metadata.json"
        write_json(meta_path, new_meta, dry_run=dry_run)

        # Move descripcion.md → ground_truth/descripcion.md
        gt_desc_dest = gt_dir / "descripcion.md"
        if not gt_desc_dest.exists():
            if dry_run:
                log.info(
                    "[DRY-RUN] Would move %s → %s",
                    desc_path.relative_to(PROJECT_ROOT),
                    gt_desc_dest.relative_to(PROJECT_ROOT),
                )
            else:
                gt_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(desc_path), str(gt_desc_dest))
                log.debug("Moved descripcion.md → ground_truth/ in %s", doc_dir.name)

        doc_count += 1

    log.info("Toledo: Normalized %d document metadata files", doc_count)

    # --- Augment catalog ---
    new_catalog = []
    for entry in catalog:
        new_entry = dict(entry)
        new_entry["doc_id"] = entry["dir_name"]
        new_entry["date"] = entry.get("slug", "")  # slug is the date
        new_entry["has_ground_truth"] = True  # all Toledo docs have transcriptions

        new_catalog.append(new_entry)

    write_json(catalog_path, new_catalog, dry_run=dry_run)
    log.info("Toledo: Augmented catalog with %d entries", len(new_catalog))


# ---------------------------------------------------------------------------
# Toledo procesal filter
# ---------------------------------------------------------------------------


def _is_procesal(letra: str | None) -> bool:
    """Return True if letra contains 'procesal' or 'procesada' (case-insensitive)."""
    if not letra:
        return False
    lower = letra.lower()
    return "procesal" in lower or "procesada" in lower


def filter_toledo_procesal(dry_run: bool = False) -> None:
    """Remove Toledo documents whose letra is not procesal/procesada.

    Keeps only documents where metadata.json → letra contains
    'procesal' or 'procesada' (case-insensitive). Updates catalog.json
    to match the remaining documents.
    """
    dataset_dir = DATA_DIR / "toledo"
    catalog_path = dataset_dir / "catalog.json"
    docs_dir = dataset_dir / "documents"

    if not catalog_path.exists():
        log.error("Toledo catalog not found at %s", catalog_path)
        return
    if not docs_dir.is_dir():
        log.error("Toledo documents directory not found at %s", docs_dir)
        return

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    keep_dirs: list[str] = []
    remove_dirs: list[str] = []

    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue

        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            log.warning("No metadata.json in %s — marking for removal", doc_dir.name)
            remove_dirs.append(doc_dir.name)
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        letra = meta.get("letra")

        if _is_procesal(letra):
            keep_dirs.append(doc_dir.name)
            log.info("KEEP   %s  (letra: %s)", doc_dir.name, letra)
        else:
            remove_dirs.append(doc_dir.name)
            log.debug("REMOVE %s  (letra: %s)", doc_dir.name, letra)

    log.info(
        "Toledo procesal filter: %d to keep, %d to remove (of %d total)",
        len(keep_dirs), len(remove_dirs), len(keep_dirs) + len(remove_dirs),
    )

    if dry_run:
        log.info("[DRY-RUN] Would remove %d directories:", len(remove_dirs))
        for d in remove_dirs:
            log.info("[DRY-RUN]   %s", d)
        log.info("[DRY-RUN] Would keep %d directories:", len(keep_dirs))
        for d in keep_dirs:
            log.info("[DRY-RUN]   %s", d)
        return

    # Remove non-procesal document directories
    for d in remove_dirs:
        target = docs_dir / d
        shutil.rmtree(target)
        log.info("Removed %s", target.relative_to(PROJECT_ROOT))

    # Update catalog to keep only matching entries
    keep_set = set(keep_dirs)
    new_catalog = [e for e in catalog if e.get("dir_name") in keep_set]
    write_json(catalog_path, new_catalog)
    log.info(
        "Toledo: Updated catalog from %d to %d entries",
        len(catalog), len(new_catalog),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset(dataset: str) -> tuple[bool, list[str]]:
    """Validate that all docs in a dataset conform to the normalized schema.

    Returns (passed, list_of_errors).
    """
    dataset_dir = DATA_DIR / dataset
    docs_dir = dataset_dir / "documents"
    catalog_path = dataset_dir / "catalog.json"
    errors: list[str] = []

    # Check catalog exists
    if not catalog_path.exists():
        errors.append(f"{dataset}: catalog.json missing")
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for entry in catalog:
            for field in ("doc_id", "dir_name", "has_ground_truth"):
                if field not in entry:
                    errors.append(f"{dataset} catalog: entry missing '{field}': {entry.get('doc_id', '?')}")

    if not docs_dir.is_dir():
        errors.append(f"{dataset}: documents/ directory missing")
        return len(errors) == 0, errors

    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_name = doc_dir.name

        # Check metadata.json exists and has core fields
        meta_path = doc_dir / "metadata.json"
        if not meta_path.exists():
            errors.append(f"{dataset}/{doc_name}: metadata.json missing")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for field in CORE_METADATA_FIELDS:
            if field not in meta:
                errors.append(f"{dataset}/{doc_name}: metadata missing '{field}'")

        # Validate ground_truth sub-object
        gt = meta.get("ground_truth", {})
        if not isinstance(gt, dict):
            errors.append(f"{dataset}/{doc_name}: ground_truth is not a dict")
        else:
            if "available" not in gt:
                errors.append(f"{dataset}/{doc_name}: ground_truth missing 'available'")
            if gt.get("type") not in GROUND_TRUTH_TYPES:
                errors.append(f"{dataset}/{doc_name}: invalid ground_truth.type: {gt.get('type')}")
            if gt.get("format") not in GROUND_TRUTH_FORMATS:
                errors.append(f"{dataset}/{doc_name}: invalid ground_truth.format: {gt.get('format')}")

        # Check facsimiles/ exists
        facs_dir = doc_dir / "facsimiles"
        if not facs_dir.is_dir():
            errors.append(f"{dataset}/{doc_name}: facsimiles/ missing")

        # Check facsimile_pages matches actual files
        listed_pages = meta.get("facsimile_pages", [])
        actual_pages = list_facsimile_pages(facs_dir)
        if sorted(listed_pages) != sorted(actual_pages):
            errors.append(
                f"{dataset}/{doc_name}: facsimile_pages mismatch — "
                f"metadata has {len(listed_pages)}, disk has {len(actual_pages)}"
            )

        # Check ground_truth/ exists
        gt_dir = doc_dir / "ground_truth"
        if not gt_dir.is_dir():
            errors.append(f"{dataset}/{doc_name}: ground_truth/ directory missing")

        # Check no descripcion.md at doc root (Toledo)
        if dataset == "toledo":
            if (doc_dir / "descripcion.md").exists():
                errors.append(f"{dataset}/{doc_name}: descripcion.md still at doc root")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Normalize dataset structures for CODEA and Toledo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --dataset all                          Normalize all datasets
  %(prog)s --dataset toledo --dry-run             Preview Toledo changes
  %(prog)s --validate --dataset all               Validate all datasets
  %(prog)s --filter-procesal --dataset toledo      Filter Toledo to procesal types
  %(prog)s --filter-procesal --dataset toledo --dry-run  Preview filter
""",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        required=True,
        help="Dataset to normalize",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset schema compliance",
    )
    parser.add_argument(
        "--filter-procesal",
        action="store_true",
        help="Filter Toledo dataset to keep only procesal/procesada letter types",
    )
    args = parser.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.filter_procesal:
        if "toledo" not in datasets:
            log.error("--filter-procesal requires --dataset toledo (or all)")
            raise SystemExit(1)
        log.info("=== Filtering Toledo to procesal letter types ===")
        filter_toledo_procesal(dry_run=args.dry_run)
        return

    if args.validate:
        all_passed = True
        for ds in datasets:
            passed, errors = validate_dataset(ds)
            if passed:
                log.info("VALIDATE %s: PASSED", ds)
            else:
                log.error("VALIDATE %s: FAILED (%d errors)", ds, len(errors))
                for err in errors:
                    log.error("  %s", err)
                all_passed = False
        if not all_passed:
            raise SystemExit(1)
        return

    handlers = {
        "codea": normalize_codea,
        "toledo": normalize_toledo,
    }

    for ds in datasets:
        log.info("=== Normalizing %s ===", ds.upper())
        handlers[ds](dry_run=args.dry_run)

    if not args.dry_run:
        log.info("=== Normalization complete. Run --validate to check. ===")


if __name__ == "__main__":
    main()
