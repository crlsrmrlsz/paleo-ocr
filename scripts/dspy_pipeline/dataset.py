"""Load and split the optimization dataset for DSPy training."""

import json
from pathlib import Path

from .config import CORPORA_DIR, OPTIMIZATION_DIR


def load_optimization_dataset() -> list[dict]:
    """Load 64 optimization pages with image paths and paleographic GT paths.

    Returns list of dicts with keys: page_id, doc_id, page, image_path, gt_path.
    All paths are relative to PROJECT_ROOT.
    """
    selection_path = OPTIMIZATION_DIR / "dataset_selection.json"
    with open(selection_path) as f:
        page_ids = json.load(f)

    entries = []
    for raw_id in page_ids:
        # raw_id format: "CODEA-3607_2r" → doc_id="CODEA-3607", page="2r"
        doc_id, page = raw_id.rsplit("_", 1)
        codea_dir = CORPORA_DIR / "codea" / "documents" / doc_id

        image_path = codea_dir / "facsimiles" / f"{page}.jpg"
        gt_path = codea_dir / "ground_truth" / f"paleographic_{page}.txt"

        entries.append({
            "page_id": f"codea_{raw_id}",  # canonical format: codea_CODEA-3607_2r
            "doc_id": doc_id,
            "page": page,
            "image_path": str(image_path.relative_to(CORPORA_DIR.parent.parent)),
            "gt_path": str(gt_path.relative_to(CORPORA_DIR.parent.parent)),
        })

    return entries


def split_train_val(
    entries: list[dict], target_train_ratio: float = 0.20
) -> tuple[list[dict], list[dict]]:
    """Document-stratified train/val split.

    Accumulates smallest documents into train until target ratio is reached.
    All pages from a document go to the same split.
    """
    # Group by document
    doc_pages: dict[str, list[dict]] = {}
    for entry in entries:
        doc_pages.setdefault(entry["doc_id"], []).append(entry)

    # Sort documents by page count (ascending)
    docs_by_size = sorted(doc_pages.items(), key=lambda x: len(x[1]))

    target_train_count = int(len(entries) * target_train_ratio)
    train_entries = []
    train_docs = set()

    for doc_id, pages in docs_by_size:
        if len(train_entries) + len(pages) > target_train_count + len(pages) // 2:
            break  # adding this doc would overshoot
        train_entries.extend(pages)
        train_docs.add(doc_id)

    # Ensure at least 2 documents in train for demo variety
    if len(train_docs) < 2:
        for doc_id, pages in docs_by_size:
            if doc_id not in train_docs:
                train_entries.extend(pages)
                train_docs.add(doc_id)
                if len(train_docs) >= 2:
                    break

    val_entries = [e for e in entries if e["doc_id"] not in train_docs]
    return train_entries, val_entries


def save_splits(train: list[dict], val: list[dict]) -> None:
    """Save train/val splits to data/optimization/splits/."""
    splits_dir = OPTIMIZATION_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train.json", "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train)} pages ({len({e['doc_id'] for e in train})} docs)")
    print(f"Val:   {len(val)} pages ({len({e['doc_id'] for e in val})} docs)")
