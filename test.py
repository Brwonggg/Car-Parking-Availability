
import pytest
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock


# ============================================================
# read_coords() — parsing logic
# ============================================================

def test_read_coords_parses_basic_format(tmp_path, monkeypatch):
    coords_file = tmp_path / "coords.txt"
    coords_file.write_text(
        "Top left:(10, 20)\n"
        "Bottom right:(30, 40)\n"
        "Top left:(50, 60)\n"
        "Bottom right:(70, 80)\n"
    )
    monkeypatch.chdir(tmp_path)
    from empty import read_coords
    result = read_coords()
    assert result == [[(10, 20), (30, 40)], [(50, 60), (70, 80)]]


def test_read_coords_empty_file_returns_empty_list(tmp_path, monkeypatch):
    coords_file = tmp_path / "coords.txt"
    coords_file.write_text("")
    monkeypatch.chdir(tmp_path)
    from empty import read_coords
    result = read_coords()
    assert result == []


# ============================================================
# coords_exist()
# ============================================================

def test_coords_exist_false_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from empty import coords_exist
    assert coords_exist() == False


def test_coords_exist_false_when_empty(tmp_path, monkeypatch):
    coords_file = tmp_path / "coords.txt"
    coords_file.write_text("")
    monkeypatch.chdir(tmp_path)
    from empty import coords_exist
    assert coords_exist() == False


def test_coords_exist_true_when_populated(tmp_path, monkeypatch):
    coords_file = tmp_path / "coords.txt"
    coords_file.write_text("Top left:(0,0)\nBottom right:(10,10)\n")
    monkeypatch.chdir(tmp_path)
    from empty import coords_exist
    assert coords_exist() == True


# ============================================================
# detect_if_empty(image, coords, device) — current 3-arg signature
# ============================================================

def test_detect_if_empty_rejects_inverted_coordinates():
    from empty import detect_if_empty
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_coords = [(50, 50), (10, 10)]  # x1 > x2, y1 > y2
    result = detect_if_empty(fake_image, invalid_coords, device=torch.device("cpu"))
    assert result is None


def test_detect_if_empty_rejects_out_of_bounds():
    from empty import detect_if_empty
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    out_of_bounds = [(10, 10), (200, 200)]
    result = detect_if_empty(fake_image, out_of_bounds, device=torch.device("cpu"))
    assert result is None


def test_detect_if_empty_rejects_zero_size_roi():
    from empty import detect_if_empty
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    zero_size = [(10, 10), (10, 50)]  # x1 == x2
    result = detect_if_empty(fake_image, zero_size, device=torch.device("cpu"))
    assert result is None


def test_detect_if_empty_returns_correct_tensor_shape():
    from empty import detect_if_empty
    fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    valid_coords = [(10, 10), (60, 60)]
    result = detect_if_empty(fake_image, valid_coords, device=torch.device("cpu"))
    assert result is not None
    assert result.shape == (1, 3, 48, 48)  # adjust if your target size differs
    assert result.dtype == torch.float32


# ============================================================
# Row-chunk interpolation math (the logic inside draw_row_chunk,
# tested standalone without the cv2 GUI interaction)
# ============================================================

def interpolate_grid(full_left, full_right, full_top, full_bottom, num_spots_per_subrow, num_sub_rows):
    """Standalone copy of the grid math from draw_row_chunk, for isolated testing."""
    spots = []
    for row_i in range(num_sub_rows):
        row_frac_start = row_i / num_sub_rows
        row_frac_end = (row_i + 1) / num_sub_rows
        y1 = int(full_top + (full_bottom - full_top) * row_frac_start)
        y2 = int(full_top + (full_bottom - full_top) * row_frac_end)
        for col_i in range(num_spots_per_subrow):
            col_frac_start = col_i / num_spots_per_subrow
            col_frac_end = (col_i + 1) / num_spots_per_subrow
            x1 = int(full_left + (full_right - full_left) * col_frac_start)
            x2 = int(full_left + (full_right - full_left) * col_frac_end)
            spots.append([(x1, y1), (x2, y2)])
    return spots


def test_interpolate_grid_produces_correct_spot_count():
    spots = interpolate_grid(0, 500, 0, 100, num_spots_per_subrow=5, num_sub_rows=2)
    assert len(spots) == 10


def test_interpolate_grid_single_subrow_matches_column_count():
    spots = interpolate_grid(0, 400, 0, 50, num_spots_per_subrow=8, num_sub_rows=1)
    assert len(spots) == 8


def test_interpolate_grid_columns_share_exact_boundaries():
    spots = interpolate_grid(0, 400, 0, 50, num_spots_per_subrow=4, num_sub_rows=1)
    for i in range(len(spots) - 1):
        assert spots[i][1][0] == spots[i + 1][0][0]  # right edge of one = left edge of next


def test_interpolate_grid_subrows_split_height_evenly():
    spots = interpolate_grid(0, 100, 0, 100, num_spots_per_subrow=1, num_sub_rows=2)
    # spots[0] is sub-row 0, spots[1] is sub-row 1 (since num_spots_per_subrow=1)
    assert spots[0][0][1] == 0
    assert spots[0][1][1] == 50
    assert spots[1][0][1] == 50
    assert spots[1][1][1] == 100


# ============================================================
# Duplicate / overlap detection (used when cleaning coords.txt)
# ============================================================

def test_remove_duplicate_coords():
    coords = [
        [(0, 0), (10, 10)],
        [(20, 20), (30, 30)],
        [(0, 0), (10, 10)],  # exact duplicate
    ]
    seen = set()
    unique = []
    for spot in coords:
        key = (tuple(spot[0]), tuple(spot[1]))
        if key not in seen:
            seen.add(key)
            unique.append(spot)
    assert len(unique) == 2


def test_boxes_overlap_detects_true_overlap():
    def boxes_overlap(box1, box2):
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    box_a = [(0, 0), (50, 50)]
    box_b = [(25, 25), (75, 75)]
    assert boxes_overlap(box_a, box_b) == True


def test_boxes_overlap_detects_no_overlap():
    def boxes_overlap(box1, box2):
        x1_1, y1_1 = box1[0]
        x2_1, y2_1 = box1[1]
        x1_2, y1_2 = box2[0]
        x2_2, y2_2 = box2[1]
        return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)

    box_a = [(0, 0), (50, 50)]
    box_c = [(100, 100), (150, 150)]
    assert boxes_overlap(box_a, box_c) == False


# ============================================================
# Model architecture sanity check
# ============================================================

def test_model_output_shape():
    from model import Model
    model = Model()
    model.eval()
    dummy_input = torch.randn(1, 3, 48, 48)
    with torch.inference_mode():
        output = model(dummy_input)
    assert output.shape == (1, 2)  # batch size 1, 2 classes


def test_model_accepts_batch_input():
    from model import Model
    model = Model()
    model.eval()
    dummy_batch = torch.randn(8, 3, 48, 48)
    with torch.inference_mode():
        output = model(dummy_batch)
    assert output.shape == (8, 2)