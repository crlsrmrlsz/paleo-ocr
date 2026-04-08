"""Region decision module — evaluates YOLO regions using page description."""

import re

import dspy


class DecideRegions(dspy.Signature):
    """You are evaluating which regions of a manuscript page to transcribe.

    For each region, decide:
    - "transcribe": main text content
    - "skip": not main text (margin note, graphic, noise, artifact)
    - "merge N": same text block split by detector

    Respond with one decision per region with a brief reason."""

    page_description: str = dspy.InputField(
        desc="Visual description of the manuscript page from PageAnalyzer"
    )
    yolo_regions: str = dspy.InputField(
        desc="YOLO detected regions with index, type, confidence%, and area%"
    )
    decisions: str = dspy.OutputField(
        desc="One decision per region: 'N: transcribe/skip/merge M — reason'"
    )


class RegionDecider(dspy.Module):
    def __init__(self):
        self.decide = dspy.ChainOfThought(DecideRegions)

    def forward(self, page_description, yolo_regions):
        return self.decide(
            page_description=page_description,
            yolo_regions=yolo_regions,
        )


def format_yolo_regions(regions, image_width, image_height):
    """Format YOLO regions as a readable string for the RegionDecider prompt."""
    page_area = image_width * image_height
    lines = []
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r.bbox
        w, h = x2 - x1, y2 - y1
        area_pct = (w * h) / page_area * 100
        lines.append(
            f"Region {i}: type={r.region_type}, confidence={r.confidence:.0%}, "
            f"position=({x1},{y1})-({x2},{y2}), size={w}x{h}, area={area_pct:.1f}%"
        )
    return "\n".join(lines)


def parse_decisions(decisions_text, n_regions):
    """Parse RegionDecider output into a list of (action, detail) tuples.

    Expected format: 'N: transcribe — reason' or 'N: skip — reason' or 'N: merge M — reason'
    Returns list of tuples: ('transcribe', ''), ('skip', 'reason'), ('merge', '2')
    Falls back to 'transcribe' for unparseable lines.
    """
    results = [("transcribe", "")] * n_regions
    for line in decisions_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?:Region\s*)?(\d+)\s*:\s*(transcribe|skip|merge\s*\d+)", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            action = m.group(2).lower().strip()
            if idx < n_regions:
                if action.startswith("merge"):
                    merge_target = re.search(r"\d+", action[5:])
                    results[idx] = ("merge", merge_target.group() if merge_target else "")
                else:
                    results[idx] = (action, "")
    return results
