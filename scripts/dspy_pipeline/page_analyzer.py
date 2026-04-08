"""Page-level visual analyzer — describes the manuscript page before transcription."""

import dspy


class AnalyzePage(dspy.Signature):
    """Describe this manuscript page visually. Do NOT attempt to read or transcribe
    any text — only describe what you see structurally:
    1. Script type and period (procesal, humanistic, etc.)
    2. Layout: where is the main text block, are there margin notes, signatures, symbols
    3. Image issues: reverse ink bleed, stains, tilt, scan artifacts, non-manuscript
       content visible (table surface, book edges)
    4. Text density and abbreviation frequency (high/medium/low)
    5. Areas that should NOT be transcribed (margin annotations, later additions)
    Do NOT mention specific names, words, or text content."""

    page_image: dspy.Image = dspy.InputField(
        desc="Full manuscript page image"
    )
    page_description: str = dspy.OutputField(
        desc="3-5 sentence structural description. NO specific text, names, or words — only layout, script type, and image quality"
    )


class PageAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzePage)

    def forward(self, page_image):
        return self.analyze(page_image=page_image)


# ---------------------------------------------------------------------------
# Split context analyzer (reading + structural)
# ---------------------------------------------------------------------------


class AnalyzePageSplitContext(dspy.Signature):
    """Analyze this manuscript page to provide context for downstream transcription
    and merging. Do NOT read or transcribe any text — only describe structural and
    visual properties."""

    page_image: dspy.Image = dspy.InputField(
        desc="Full manuscript page image"
    )
    reading_context: str = dspy.OutputField(
        desc="Page-level information NOT visible in individual cropped strips: "
             "whether faint text is bleed-through from the verso side (not content "
             "to transcribe), damage or stain patterns that span the page, whether "
             "margin annotations intrude into the main text block, page-edge "
             "artifacts (binding shadow, table surface, book edges). 2-3 sentences."
    )
    structural_context: str = dspy.OutputField(
        desc="For merging overlapping strips: layout structure (single column, "
             "two columns, margin notes), text flow direction, section breaks, "
             "signatures or rubrics, page completeness (full page or fragment). "
             "2-3 sentences."
    )


class SplitContextAnalyzer(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzePageSplitContext)

    def forward(self, page_image):
        return self.analyze(page_image=page_image)


# ---------------------------------------------------------------------------
# Noise detector (anti-hallucination)
# ---------------------------------------------------------------------------


class AnalyzePageNoise(dspy.Signature):
    """Identify visual noise in this manuscript image that could be mistaken for text:
    - Bleed-through: faint text from the reverse side of the page
    - Shadows: binding shadow, fold darkness
    - Stains or damage that resemble ink strokes
    Answer in 1-2 sentences. If the page is clean, say "clean page"."""

    page_image: dspy.Image = dspy.InputField(
        desc="Cropped manuscript page image"
    )
    noise_warnings: str = dspy.OutputField(
        desc="Visual noise warnings: what marks in the image are NOT real text "
             "(bleed-through from reverse, shadows, stains). 1-2 sentences."
    )


class NoiseDetector(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzePageNoise)

    def forward(self, page_image):
        return self.analyze(page_image=page_image)
