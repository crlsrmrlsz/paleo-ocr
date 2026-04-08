"""Tests for DSPy HTR pipeline components."""

import types

import pytest


class TestCerMetric:
    """Test the DSPy-compatible CER metric function."""

    @pytest.fixture
    def cer_metric(self):
        from scripts.dspy_pipeline.metric import cer_metric
        return cer_metric

    def _make_example(self, gt: str):
        return types.SimpleNamespace(transcription=gt)

    def _make_prediction(self, pred: str):
        return types.SimpleNamespace(transcription=pred)

    def test_perfect_match(self, cer_metric):
        ex = self._make_example("hello world")
        pred = self._make_prediction("hello world")
        assert cer_metric(ex, pred) == 1.0

    def test_complete_mismatch(self, cer_metric):
        ex = self._make_example("abc")
        pred = self._make_prediction("xyz")
        result = cer_metric(ex, pred)
        assert result == pytest.approx(0.0)

    def test_partial_error(self, cer_metric):
        ex = self._make_example("abcdef")
        pred = self._make_prediction("abcxyz")
        result = cer_metric(ex, pred)
        assert result == pytest.approx(0.5)

    def test_empty_gt_empty_pred(self, cer_metric):
        ex = self._make_example("")
        pred = self._make_prediction("")
        assert cer_metric(ex, pred) == 1.0

    def test_empty_gt_nonempty_pred(self, cer_metric):
        ex = self._make_example("")
        pred = self._make_prediction("something")
        assert cer_metric(ex, pred) == 0.0

    def test_longer_hypothesis_can_exceed_one(self, cer_metric):
        """CER can exceed 1.0 when hypothesis is much longer; metric floors at 0.0."""
        ex = self._make_example("a")
        pred = self._make_prediction("abcde")
        result = cer_metric(ex, pred)
        assert result == 0.0

    def test_whitespace_normalization(self, cer_metric):
        """Newlines and multi-space are collapsed before CER computation."""
        ex = self._make_example("d\nic\nho")  # GT multi-line abbreviation
        pred = self._make_prediction("dicho")  # model output as one word
        result = cer_metric(ex, pred)
        # "d ic ho" vs "dicho" — NOT penalized for newlines
        assert result > 0.0  # not a complete mismatch
        assert result < 1.0  # still some edit distance (spaces)

    def test_gepa_5arg_signature(self, cer_metric):
        """GEPA calls with 5 args; returns ScoreWithFeedback when pred_name set."""
        ex = self._make_example("hello world")
        pred = self._make_prediction("hello world")
        result = cer_metric(ex, pred, None, "transcribe", None)
        assert hasattr(result, "score")
        assert hasattr(result, "feedback")
        assert result.score == 1.0

    def test_gepa_feedback_content(self, cer_metric):
        """Feedback text varies by score level."""
        ex = self._make_example("abcdef")
        pred = self._make_prediction("xyzxyz")
        result = cer_metric(ex, pred, None, "transcribe", None)
        assert result.score == pytest.approx(0.0)
        assert "High error rate" in result.feedback


class TestDatasetLoader:
    """Test optimization dataset loading and splitting."""

    @pytest.fixture
    def load_optimization_dataset(self):
        from scripts.dspy_pipeline.dataset import load_optimization_dataset
        return load_optimization_dataset

    @pytest.fixture
    def split_train_val(self):
        from scripts.dspy_pipeline.dataset import split_train_val
        return split_train_val

    def test_load_returns_entries_with_required_fields(self, load_optimization_dataset):
        entries = load_optimization_dataset()
        assert len(entries) == 64
        for entry in entries:
            assert "page_id" in entry
            assert "doc_id" in entry
            assert "image_path" in entry
            assert "gt_path" in entry

    def test_all_image_paths_exist(self, load_optimization_dataset):
        from scripts.dspy_pipeline.config import PROJECT_ROOT
        entries = load_optimization_dataset()
        for entry in entries:
            path = PROJECT_ROOT / entry["image_path"]
            assert path.exists(), f"Missing image: {entry['image_path']}"

    def test_all_gt_paths_exist(self, load_optimization_dataset):
        from scripts.dspy_pipeline.config import PROJECT_ROOT
        entries = load_optimization_dataset()
        for entry in entries:
            path = PROJECT_ROOT / entry["gt_path"]
            assert path.exists(), f"Missing GT: {entry['gt_path']}"

    def test_split_is_document_stratified(self, load_optimization_dataset, split_train_val):
        entries = load_optimization_dataset()
        train, val = split_train_val(entries)
        train_docs = {e["doc_id"] for e in train}
        val_docs = {e["doc_id"] for e in val}
        assert train_docs.isdisjoint(val_docs), "Train and val share documents"

    def test_split_covers_all_entries(self, load_optimization_dataset, split_train_val):
        entries = load_optimization_dataset()
        train, val = split_train_val(entries)
        assert len(train) + len(val) == len(entries)

    def test_train_is_roughly_20_percent(self, load_optimization_dataset, split_train_val):
        entries = load_optimization_dataset()
        train, val = split_train_val(entries)
        ratio = len(train) / len(entries)
        assert 0.10 <= ratio <= 0.30, f"Train ratio {ratio:.2f} outside 10-30%"


class TestTranscriberModule:
    """Test DSPy module structure (no LLM calls)."""

    def test_signature_has_correct_fields(self):
        from scripts.dspy_pipeline.transcriber import TranscribeManuscript
        fields = TranscribeManuscript.model_fields
        assert "page_image" in fields
        assert "transcription" in fields

    def test_module_has_chain_of_thought(self):
        from scripts.dspy_pipeline.transcriber import ManuscriptTranscriber
        module = ManuscriptTranscriber()
        assert hasattr(module, "transcribe")

    def test_module_is_dspy_module(self):
        import dspy
        from scripts.dspy_pipeline.transcriber import ManuscriptTranscriber
        module = ManuscriptTranscriber()
        assert isinstance(module, dspy.Module)
