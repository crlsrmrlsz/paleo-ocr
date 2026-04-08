# tests/test_tile_grid_split.py
"""Tests for 2D tile grid splitting."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.dspy_pipeline.block_pipeline import tile_grid_split


@pytest.fixture
def make_image(tmp_path):
    """Create a test image of given dimensions."""
    def _make(width, height, color="white"):
        img = Image.new("RGB", (width, height), color)
        path = tmp_path / f"test_{width}x{height}.jpg"
        img.save(str(path), format="JPEG")
        img.close()
        return str(path)
    return _make


class TestTileGridSplit:
    def test_single_tile_when_fits(self, make_image):
        """Image smaller than max_px on both axes → 1 tile, no split."""
        path = make_image(1000, 1200)
        grid = tile_grid_split(path, (0, 0, 1000, 1200), max_px=1568)
        assert len(grid) == 1  # 1 row
        assert len(grid[0]) == 1  # 1 col
        assert grid[0][0].crop_path is not None

    def test_vertical_only_split(self, make_image):
        """Width fits, height exceeds → 1 col, multiple rows."""
        path = make_image(1200, 3500)
        grid = tile_grid_split(path, (0, 0, 1200, 3500), max_px=1568)
        assert len(grid) >= 2  # multiple rows
        assert all(len(row) == 1 for row in grid)  # single column

    def test_horizontal_only_split(self, make_image):
        """Width exceeds, height fits → 1 row, multiple cols."""
        path = make_image(3500, 1200)
        grid = tile_grid_split(path, (0, 0, 3500, 1200), max_px=1568)
        assert len(grid) == 1  # 1 row
        assert len(grid[0]) >= 2  # multiple columns

    def test_2d_split(self, make_image):
        """Both axes exceed → 2D grid."""
        path = make_image(2700, 4000)
        grid = tile_grid_split(path, (0, 0, 2700, 4000), max_px=1568)
        assert len(grid) >= 2  # multiple rows
        assert len(grid[0]) >= 2  # multiple cols

    def test_tile_dimensions_within_limit(self, make_image):
        """Every tile must fit within max_px on both axes."""
        path = make_image(2700, 4000)
        grid = tile_grid_split(path, (0, 0, 2700, 4000), max_px=1568)
        for row in grid:
            for tile in row:
                x1, y1, x2, y2 = tile.bbox
                assert x2 - x1 <= 1568
                assert y2 - y1 <= 1568

    def test_content_bbox_offset(self, make_image):
        """Content bbox with offset — tiles should start from bbox origin."""
        path = make_image(3000, 4500)
        grid = tile_grid_split(path, (100, 200, 2800, 4200), max_px=1568)
        # First tile starts at content origin
        assert grid[0][0].bbox[0] == 100
        assert grid[0][0].bbox[1] == 200

    def test_tiny_trailing_absorbed(self, make_image):
        """Trailing sliver (<20% max_px) absorbed into previous tile."""
        path = make_image(1600, 1568)
        grid = tile_grid_split(path, (0, 0, 1600, 1568), max_px=1568)
        assert len(grid) == 1
        assert len(grid[0]) == 1

    def test_crop_paths_exist(self, make_image):
        """All tiles have crop_path pointing to an existing JPEG."""
        path = make_image(2700, 4000)
        grid = tile_grid_split(path, (0, 0, 2700, 4000), max_px=1568)
        for row in grid:
            for tile in row:
                assert tile.crop_path is not None
                assert Path(tile.crop_path).exists()

    def test_overlap_between_adjacent_tiles(self, make_image):
        """Adjacent tiles overlap by the specified amount."""
        path = make_image(2700, 4000)
        grid = tile_grid_split(path, (0, 0, 2700, 4000), max_px=1568, overlap=200)
        # Check vertical overlap between row 0 and row 1
        if len(grid) >= 2:
            r0_bottom = grid[0][0].bbox[3]  # y2 of first row
            r1_top = grid[1][0].bbox[1]  # y1 of second row
            assert r0_bottom - r1_top == 200
        # Check horizontal overlap between col 0 and col 1
        if len(grid[0]) >= 2:
            c0_right = grid[0][0].bbox[2]  # x2 of first col
            c1_left = grid[0][1].bbox[0]  # x1 of second col
            assert c0_right - c1_left == 200
