"""Tests for vocabulary-guided correction."""

import pytest
from scripts.dspy_pipeline.vocab_correction import (
    load_combined_vocab,
    load_abbreviations,
    find_candidates,
    find_compound_split,
    find_abbrev_expansion,
    flag_unknown_words,
)


class TestLoadData:
    def test_load_combined_vocab(self):
        vocab = load_combined_vocab()
        assert len(vocab) > 90000
        assert "dicho" in vocab
        assert "çenso" in vocab
        assert "maravedis" in vocab

    def test_load_abbreviations(self):
        abbrevs = load_abbreviations()
        assert "dho" in abbrevs
        assert abbrevs["dho"] == "dicho"
        assert "q" in abbrevs
        assert len(abbrevs) > 100


class TestFindCandidates:
    @pytest.fixture
    def vocab(self):
        return load_combined_vocab()

    def test_edit_distance_1(self, vocab):
        candidates = find_candidates("censp", vocab, max_dist=3, top_n=3)
        assert any(c[0] == "censo" for c in candidates)
        assert candidates[0][1] <= 1

    def test_no_candidates_for_garbage(self, vocab):
        candidates = find_candidates("zzzzzzzzz", vocab, max_dist=3, top_n=3)
        assert len(candidates) == 0

    def test_top_n_limit(self, vocab):
        candidates = find_candidates("dich", vocab, max_dist=3, top_n=2)
        assert len(candidates) <= 2

    def test_max_dist_filter(self, vocab):
        candidates = find_candidates("dich", vocab, max_dist=1, top_n=5)
        assert all(c[1] <= 1 for c in candidates)


class TestCompoundSplit:
    @pytest.fixture
    def vocab(self):
        return load_combined_vocab()

    def test_simple_compound(self, vocab):
        result = find_compound_split("deladicha", vocab)
        assert result is not None
        assert all(part in vocab for part in result)

    def test_no_split_for_garbage(self, vocab):
        result = find_compound_split("zzzzz", vocab)
        assert result is None

    def test_min_part_length(self, vocab):
        result = find_compound_split("abc", vocab)
        if result is not None:
            assert all(len(part) >= 2 for part in result)


class TestAbbrevExpansion:
    @pytest.fixture
    def vocab(self):
        return load_combined_vocab()

    @pytest.fixture
    def abbrevs(self):
        return load_abbreviations()

    def test_dho_to_dicho(self, vocab, abbrevs):
        result = find_abbrev_expansion("dho", vocab, abbrevs)
        assert result is not None
        assert "dicho" in result

    def test_no_expansion_for_garbage(self, vocab, abbrevs):
        result = find_abbrev_expansion("zzzzz", vocab, abbrevs)
        assert result is None or len(result) == 0


class TestFlagUnknownWords:
    @pytest.fixture
    def vocab(self):
        return load_combined_vocab()

    def test_flags_unknown(self, vocab):
        text = "dicho censo zzzzgarbage otra"
        flagged = flag_unknown_words(text, vocab)
        assert any(f["word"] == "zzzzgarbage" for f in flagged)
        assert not any(f["word"] == "dicho" for f in flagged)

    def test_skips_short_words(self, vocab):
        text = "a e y zzzzgarbage"
        flagged = flag_unknown_words(text, vocab)
        assert not any(f["word"] in ("a", "e", "y") for f in flagged)

    def test_preserves_position(self, vocab):
        text = "dicho zzzzgarbage otra"
        flagged = flag_unknown_words(text, vocab)
        assert len(flagged) >= 1
        assert "start" in flagged[0]
        assert "end" in flagged[0]
