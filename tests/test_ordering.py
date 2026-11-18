"""Tests for the grouped tile ordering model in gemm/ordering.py.

Author: Kajal Changappa

Pure Python, so they run everywhere. The key property: whatever the matrix
and group size, every tile of C is computed by exactly one program. A tile
visited twice wastes work; a tile never visited leaves garbage in C.
"""
import pytest

from gemm.ordering import (
    cdiv,
    distinct_tiles_loaded,
    grouped_order,
    grouped_tile,
    row_major_tile,
)


@pytest.mark.parametrize("num_pid_m,num_pid_n", [(1, 1), (4, 4), (5, 3), (7, 2), (3, 9), (16, 16), (1, 10), (10, 1)])
@pytest.mark.parametrize("group_m", [1, 2, 3, 8])
def test_every_tile_visited_exactly_once(num_pid_m, num_pid_n, group_m):
    tiles = list(grouped_order(num_pid_m, num_pid_n, group_m))
    assert len(tiles) == num_pid_m * num_pid_n
    assert set(tiles) == {(m, n) for m in range(num_pid_m) for n in range(num_pid_n)}


def test_matches_the_documented_4x4_example():
    # The picture in gemm/ordering.py, GROUP_M = 2.
    expected = [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]
    for m in range(4):
        for n in range(4):
            assert grouped_tile(expected[m][n], 4, 4, 2) == (m, n)


def test_group_of_one_is_row_major():
    for pid in range(5 * 7):
        assert grouped_tile(pid, 5, 7, 1) == row_major_tile(pid, 7)


def test_last_group_can_be_partial():
    # 5 tile rows with GROUP_M = 2 leaves a final group of 1 row.
    last_group = [grouped_tile(pid, 5, 3, 2) for pid in range(12, 15)]
    assert last_group == [(4, 0), (4, 1), (4, 2)]


def test_grouping_needs_fewer_distinct_tiles():
    # 16 programs running together on a 16 x 16 tile grid.
    row_major = [row_major_tile(pid, 16) for pid in range(16)]
    grouped = [grouped_tile(pid, 16, 16, 4) for pid in range(16)]
    assert distinct_tiles_loaded(row_major) == (1, 16)  # 17 tiles in total
    assert distinct_tiles_loaded(grouped) == (4, 4)     # 8 tiles in total


@pytest.mark.parametrize("a,b,expected", [(0, 4, 0), (1, 4, 1), (4, 4, 1), (5, 4, 2), (130, 64, 3)])
def test_cdiv(a, b, expected):
    assert cdiv(a, b) == expected
