"""DSPy signature and module for manuscript transcription."""

import dspy

from .abbreviations import READING_REFERENCE


# --- Baseline: minimal prompt, no reference ---


class TranscribeManuscript(dspy.Signature):
    """Transcribe a historical Spanish manuscript page image. Preserve original
    spelling exactly — do NOT modernize. Expand abbreviation marks. Transcribe
    only what is visible on the page.
    """

    page_image: dspy.Image = dspy.InputField(
        desc="Facsimile image of a manuscript page"
    )
    transcription: str = dspy.OutputField(
        desc="Paleographic transcription preserving original spelling, "
        "with abbreviations expanded"
    )


class ManuscriptTranscriber(dspy.Module):
    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeManuscript)

    def forward(self, page_image):
        return self.transcribe(page_image=page_image)


# --- Step D.2: contextual — reference data as InputField ---


class TranscribeWithReference(dspy.Signature):
    """Transcribe a historical Spanish manuscript page image. Preserve original
    spelling exactly — do NOT modernize. Expand abbreviation marks. Transcribe
    only what is visible on the page. Consult the reference guide for help
    recognizing abbreviation marks and period spellings.
    """

    page_image: dspy.Image = dspy.InputField(
        desc="Facsimile image of a manuscript page"
    )
    reference: str = dspy.InputField(
        desc="Visual guide to abbreviation marks, common abbreviated forms, "
        "and period spellings found in these manuscripts"
    )
    transcription: str = dspy.OutputField(
        desc="Paleographic transcription preserving original spelling, "
        "with abbreviations expanded"
    )


class ContextualManuscriptTranscriber(dspy.Module):
    """Same as ManuscriptTranscriber but passes reading reference
    as structured data alongside the image."""

    def __init__(self):
        self.transcribe = dspy.ChainOfThought(TranscribeWithReference)

    def forward(self, page_image):
        return self.transcribe(
            page_image=page_image,
            reference=READING_REFERENCE,
        )
