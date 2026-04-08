"""DSPy module for per-block manuscript transcription."""

import dspy


class TranscribeBlock(dspy.Signature):
    """Transcribe a cropped region from a 16th-century Spanish notarial manuscript page
    (procesal script). Preserve original spelling exactly: u/v, i/j, ç/z, x/j, ss/s as
    written in the manuscript. Expand abbreviations silently (dicho not d<ic>ho). Do NOT
    modernize (dixo not dijo, conoçe not conoce). Preserve word boundaries as written
    (deladicha, quese). For graphic/stamp regions, output the appropriate marker:
    [rúbrica], [cruz], [signo]."""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    region_type: str = dspy.InputField(
        desc="Type of region: main_text, margin, graphic, stamp"
    )
    transcription: str = dspy.OutputField(
        desc="Paleographic transcription preserving original spelling, abbreviations expanded silently"
    )


class BlockTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeBlock)

    def forward(self, block_image, region_type):
        return self.transcribe(
            block_image=block_image,
            region_type=region_type,
        )


class MergeStripTranscriptions(dspy.Signature):
    """Merge overlapping strip transcriptions from a 16th-century Spanish manuscript into
    a single coherent transcription. Adjacent strips share ~3-5 lines of overlapping text.
    Deduplicate the overlapping text. Preserve original spelling exactly: u/v, i/j, ç/z,
    x/j, ss/s as written. Do NOT modernize. Keep abbreviations expanded silently."""

    strip_transcriptions: str = dspy.InputField(
        desc="High-resolution strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    region_type: str = dspy.InputField(
        desc="Type of region these strips came from: main_text, margin"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, preserving original spelling"
    )


class StripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripTranscriptions)

    def forward(self, strip_transcriptions, region_type):
        return self.merge(
            strip_transcriptions=strip_transcriptions,
            region_type=region_type,
        )


# ---------------------------------------------------------------------------
# Enriched variants — same signatures + domain knowledge InputFields
# ---------------------------------------------------------------------------


class TranscribeBlockEnriched(dspy.Signature):
    """Transcribe a cropped region from a historical Spanish manuscript page, using domain knowledge to resolve abbreviations and character ambiguities."""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    region_type: str = dspy.InputField(
        desc="Type of region: main_text, margin, graphic, stamp"
    )
    full_page_transcription: str = dspy.InputField(
        desc="Rough transcription of the full page for context"
    )
    abbreviation_table: str = dspy.InputField(
        desc="JSON dictionary of common procesal abbreviations: abbreviated form → expanded form"
    )
    confusion_pairs: str = dspy.InputField(
        desc="Confusable letter pairs in procesal script with visual distinguishing features"
    )
    valid_alternations: str = dspy.InputField(
        desc="Period-standard spelling alternations (XVI-XVII century) that must be preserved as written"
    )
    transcription: str = dspy.OutputField(
        desc="Paleographic transcription of this region, preserving original spelling, "
        "with abbreviations expanded. For graphic/stamp regions, output the "
        "appropriate marker: [rúbrica], [cruz], [signo]"
    )


class EnrichedBlockTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeBlockEnriched)

    def forward(
        self,
        block_image,
        region_type,
        full_page_transcription,
        abbreviation_table="",
        confusion_pairs="",
        valid_alternations="",
    ):
        return self.transcribe(
            block_image=block_image,
            region_type=region_type,
            full_page_transcription=full_page_transcription,
            abbreviation_table=abbreviation_table,
            confusion_pairs=confusion_pairs,
            valid_alternations=valid_alternations,
        )


class MergeStripTranscriptionsEnriched(dspy.Signature):
    """Merge overlapping strip transcriptions using domain knowledge.

    Multiple overlapping horizontal strips were cropped from the same text region
    and transcribed independently at high resolution. Adjacent strips share ~3-5
    lines of overlapping text that appears in both. The full-page transcription
    (lower resolution but structurally complete) serves as a reference for the
    overall text flow and structure.

    Produce a single merged transcription that:
    - Deduplicates text that appears in multiple strips (overlap regions)
    - Preserves the high-resolution readings from individual strips
    - Follows the structural flow of the full-page transcription
    - Keeps original spelling and expanded abbreviations
    - Uses domain knowledge to resolve ambiguities in overlap regions
    """

    strip_transcriptions: str = dspy.InputField(
        desc="All strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    full_page_transcription: str = dspy.InputField(
        desc="Complete rough transcription of the full page (lower resolution, for structural reference)"
    )
    region_type: str = dspy.InputField(
        desc="Type of region these strips came from: main_text, margin"
    )
    abbreviation_table: str = dspy.InputField(
        desc="JSON dictionary of common procesal abbreviations: abbreviated form → expanded form"
    )
    confusion_pairs: str = dspy.InputField(
        desc="Confusable letter pairs in procesal script with visual distinguishing features"
    )
    valid_alternations: str = dspy.InputField(
        desc="Period-standard spelling alternations (XVI-XVII century) that must be preserved as written"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged paleographic transcription with overlapping text deduplicated, "
        "preserving original spelling and abbreviation expansions from the high-res strips"
    )


class EnrichedStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripTranscriptionsEnriched)

    def forward(
        self,
        strip_transcriptions,
        full_page_transcription,
        region_type,
        abbreviation_table="",
        confusion_pairs="",
        valid_alternations="",
    ):
        return self.merge(
            strip_transcriptions=strip_transcriptions,
            full_page_transcription=full_page_transcription,
            region_type=region_type,
            abbreviation_table=abbreviation_table,
            confusion_pairs=confusion_pairs,
            valid_alternations=valid_alternations,
        )


# ---------------------------------------------------------------------------
# Standard strip transcription
# ---------------------------------------------------------------------------


class TranscribeStrip(dspy.Signature):
    """Transcribe the main text in this manuscript strip.

    - Expand abbreviations silently: dho → dicho, q → que (no angle brackets)
    - Preserve original spelling: u/v, i/j, ç/z, x/j, ss/s as written
    - Do NOT modernize: dixo stays dixo (not dijo)
    - Preserve word boundaries as written: deladicha, quese
    - Preserve capitalization as it appears in the image
    - Output plain text only, no markers or brackets"""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    transcription: str = dspy.OutputField(
        desc="Plain text transcription, original spelling and capitalization preserved, abbreviations expanded silently"
    )


class StripTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeStrip)

    def forward(self, block_image, **kwargs):
        return self.transcribe(block_image=block_image)


# ---------------------------------------------------------------------------
# Context-aware strip transcription (reading_context)
# ---------------------------------------------------------------------------


class TranscribeStripWithContext(dspy.Signature):
    """Transcribe the main text in this manuscript strip.

    - Expand abbreviations silently: dho → dicho, q → que (no angle brackets)
    - Preserve original spelling: u/v, i/j, ç/z, x/j, ss/s as written
    - Do NOT modernize: dixo stays dixo (not dijo)
    - Preserve word boundaries as written: deladicha, quese
    - Preserve capitalization as it appears in the image
    - Output plain text only, no markers or brackets"""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    reading_context: str = dspy.InputField(
        desc="Page-level context not visible in this strip: bleed-through patterns, "
             "page damage, margin intrusions, edge artifacts"
    )
    transcription: str = dspy.OutputField(
        desc="Plain text transcription, original spelling and capitalization preserved, "
             "abbreviations expanded silently"
    )


class ContextAwareStripTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeStripWithContext)

    def forward(self, block_image, reading_context="", **kwargs):
        return self.transcribe(
            block_image=block_image,
            reading_context=reading_context,
        )


class MergeStripsWithStructure(dspy.Signature):
    """Merge overlapping strip transcriptions from a manuscript page into a single text.
    Adjacent strips share ~3-5 lines. Deduplicate the overlap.
    Preserve original spelling and capitalization. Plain text only."""

    strip_transcriptions: str = dspy.InputField(
        desc="Strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    structural_context: str = dspy.InputField(
        desc="Page layout and structure: columns, text flow, section breaks, "
             "signatures, page completeness"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, original "
             "spelling and capitalization preserved"
    )


class StructuralStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripsWithStructure)

    def forward(self, strip_transcriptions, structural_context=""):
        return self.merge(
            strip_transcriptions=strip_transcriptions,
            structural_context=structural_context,
        )


# ---------------------------------------------------------------------------
# Strip merger with page description
# ---------------------------------------------------------------------------


class MergeStripsWithPageDesc(dspy.Signature):
    """Merge overlapping strip transcriptions from a manuscript page into a single text.
    Adjacent strips share ~3-5 lines. Deduplicate the overlap.
    Preserve original spelling. Plain text only."""

    strip_transcriptions: str = dspy.InputField(
        desc="Strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    page_description: str = dspy.InputField(
        desc="Structural description of the page: script type, layout, image quality"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, original spelling preserved"
    )


class PageDescStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripsWithPageDesc)

    def forward(self, strip_transcriptions, page_description=""):
        return self.merge(
            strip_transcriptions=strip_transcriptions,
            page_description=page_description,
        )


# ---------------------------------------------------------------------------
# Noise-aware strip merger
# ---------------------------------------------------------------------------


class MergeStripsNoiseAware(dspy.Signature):
    """Merge overlapping strip transcriptions from a manuscript page into a single text.
    Adjacent strips share ~3-5 lines. Deduplicate the overlap.
    Discard any text that matches the noise warnings (bleed-through, shadows, stains).
    Preserve original spelling and capitalization. Plain text only."""

    strip_transcriptions: str = dspy.InputField(
        desc="Strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    noise_warnings: str = dspy.InputField(
        desc="Visual noise warnings: what marks in the image are NOT real text "
             "(bleed-through from reverse, shadows, stains)"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, noise-induced "
             "text discarded, original spelling and capitalization preserved"
    )


class NoiseAwareStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripsNoiseAware)

    def forward(self, strip_transcriptions, noise_warnings=""):
        return self.merge(
            strip_transcriptions=strip_transcriptions,
            noise_warnings=noise_warnings,
        )


# ---------------------------------------------------------------------------
# Anti-hallucination strip transcriber
# ---------------------------------------------------------------------------


class TranscribeStripAntiHalluc(dspy.Signature):
    """Transcribe the main text in this manuscript strip.

    - Expand abbreviations silently: dho → dicho, q → que (no angle brackets)
    - Preserve original spelling: u/v, i/j, ç/z, x/j, ss/s as written
    - Do NOT modernize: dixo stays dixo (not dijo)
    - Preserve word boundaries as written: deladicha, quese
    - Preserve capitalization as it appears in the image
    - DO NOT INSERT WORDS: if a passage is unclear, transcribe only what you
      can see. Do NOT add function words (de, que, en, el, y, e, por, vn,
      vna, la, los) to improve grammar or fill gaps. Procesal text frequently
      omits connectors — leave gaps as they appear.
    - Output plain text only, no markers or brackets"""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    transcription: str = dspy.OutputField(
        desc="Plain text transcription, original spelling and capitalization preserved, "
             "abbreviations expanded silently, no invented words"
    )


class AntiHallucStripTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeStripAntiHalluc)

    def forward(self, block_image, **kwargs):
        return self.transcribe(block_image=block_image)


# ---------------------------------------------------------------------------
# Anti-deletion strip merger
# ---------------------------------------------------------------------------


class MergeStripsAntiDelete(dspy.Signature):
    """Merge overlapping strip transcriptions from a manuscript page into a single text.
    Adjacent strips share ~3-5 lines of overlapping text. Deduplicate the overlap.

    When two strips give different readings of the same passage, prefer the reading
    that forms more complete, recognizable words. Do NOT delete text that appears
    in only one strip — include it even if it looks unusual.

    Preserve original spelling and capitalization. Plain text only."""

    strip_transcriptions: str = dspy.InputField(
        desc="Strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, "
             "original spelling and capitalization preserved"
    )


# ---------------------------------------------------------------------------
# Forced visual reasoning transcriber
# ---------------------------------------------------------------------------


class TranscribeStripVisualReasoning(dspy.Signature):
    """Transcribe the main text in this manuscript strip.

    - Expand abbreviations silently: dho → dicho, q → que (no angle brackets)
    - Preserve original spelling: u/v, i/j, ç/z, x/j, ss/s as written
    - Do NOT modernize: dixo stays dixo (not dijo)
    - Preserve word boundaries as written: deladicha, quese
    - Preserve capitalization as it appears in the image
    - Output plain text only, no markers or brackets"""

    block_image: dspy.Image = dspy.InputField(
        desc="Cropped image of a manuscript region at high resolution"
    )
    reasoning: str = dspy.OutputField(
        desc="Line-by-line visual analysis: for each line, describe the letterforms "
             "and strokes you see, identify abbreviation marks (tildes, superscripts, "
             "loops), note ambiguous characters (u/v, i/j, c/ç, s/ss, x/j) and explain "
             "how you resolve them. Flag any passages where the ink is faded, "
             "bleed-through obscures text, or strokes are unclear."
    )
    transcription: str = dspy.OutputField(
        desc="Plain text transcription, original spelling and capitalization preserved, "
             "abbreviations expanded silently"
    )


class VisualReasoningStripTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.Predict(TranscribeStripVisualReasoning)

    def forward(self, block_image, **kwargs):
        return self.transcribe(block_image=block_image)


class AntiDeleteStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripsAntiDelete)

    def forward(self, strip_transcriptions, **kwargs):
        return self.merge(strip_transcriptions=strip_transcriptions)


# ---------------------------------------------------------------------------
# Bare strip merger (minimal)
# ---------------------------------------------------------------------------


class MergeStripsBare(dspy.Signature):
    """Merge overlapping strip transcriptions from a manuscript page into a single text.
    Adjacent strips share ~3-5 lines of overlapping text. Deduplicate the overlap.
    Preserve original spelling and capitalization. Plain text only."""

    strip_transcriptions: str = dspy.InputField(
        desc="Strip transcriptions in order, separated by '---STRIP N---' headers"
    )
    transcription: str = dspy.OutputField(
        desc="Single merged transcription with overlap deduplicated, "
             "original spelling and capitalization preserved"
    )


class BareStripMerger(dspy.Module):
    def __init__(self):
        self.merge = dspy.ChainOfThought(MergeStripsBare)

    def forward(self, strip_transcriptions, **kwargs):
        return self.merge(strip_transcriptions=strip_transcriptions)
