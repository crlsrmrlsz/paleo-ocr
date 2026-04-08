"""Page-level critic — reviews merged transcription with an independent model.

Uses a different model from the transcriber (e.g., GPT-5.4) to avoid
confirming the transcriber's own visual perception errors.
"""

import dspy


class ReviewTranscription(dspy.Signature):
    """Review a paleographic transcription of a historical Spanish manuscript page.
    Identify inconsistencies, potential errors, and strips that should be reprocessed."""

    merged_transcription: str = dspy.InputField(
        desc="Merged transcription of all strips from this page"
    )
    rough_transcription: str = dspy.InputField(
        desc="Initial rough transcription of the full page (lower resolution reference)"
    )
    page_context: str = dspy.InputField(
        desc="Context accumulated from previous pages and strips: abbreviation patterns, names, places"
    )
    review: str = dspy.OutputField(
        desc="Analysis of transcription quality: inconsistencies, suspected errors, confidence assessment"
    )
    corrected_transcription: str = dspy.OutputField(
        desc="Corrected transcription with fixes applied. If no fixes needed, return the original."
    )
    strips_to_reprocess: str = dspy.OutputField(
        desc="Comma-separated strip indices (0-based) that need reprocessing, or 'none'"
    )


class Critic(dspy.Module):
    def __init__(self, critic_lm=None):
        self.review = dspy.ChainOfThought(ReviewTranscription)
        self.critic_lm = critic_lm

    def forward(self, merged_transcription, rough_transcription, page_context=""):
        if self.critic_lm:
            with dspy.context(lm=self.critic_lm):
                return self.review(
                    merged_transcription=merged_transcription,
                    rough_transcription=rough_transcription,
                    page_context=page_context,
                )
        return self.review(
            merged_transcription=merged_transcription,
            rough_transcription=rough_transcription,
            page_context=page_context,
        )
