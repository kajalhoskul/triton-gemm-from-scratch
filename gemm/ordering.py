"""Tile ordering: which tile of C each program computes, and in what order.

Author: Kajal Changappa

GPUs run programs roughly in order of their program id. In the step 3
kernel, ids map to tiles row by row:

    row-major, 4 x 4 tiles          grouped, GROUP_M = 2
     0  1  2  3                       0  2  4  6
     4  5  6  7                       1  3  5  7
     8  9 10 11                       8 10 12 14
    12 13 14 15                       9 11 13 15

Programs running at the same time share the L2 cache. In row-major order,
programs 0-3 need one row of A tiles but ALL FOUR columns of B tiles. In
grouped order, programs 0-3 need two rows of A and only two columns of B: the
same amount of work from fewer distinct tiles, so more of what they load is
already in L2. On large matrices this cuts global memory traffic noticeably.

This module is the plain-Python model of the mapping. The Triton kernel in
gemm/fast.py does the same arithmetic with tl operations, and the tests use
this model to prove the mapping visits every tile exactly once.
"""
from typing import Iterator, Tuple


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def grouped_tile(pid: int, num_pid_m: int, num_pid_n: int, group_m: int) -> Tuple[int, int]:
    """(tile row, tile column) for program `pid` in grouped order.

    Tiles are handed out in groups of `group_m` tile rows. Inside a group,
    ids go down a column first, then move to the next column. The last group
    may have fewer than `group_m` rows.
    """
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def row_major_tile(pid: int, num_pid_n: int) -> Tuple[int, int]:
    """(tile row, tile column) for program `pid` in the step 3 order."""
    return pid // num_pid_n, pid % num_pid_n


def grouped_order(num_pid_m: int, num_pid_n: int, group_m: int) -> Iterator[Tuple[int, int]]:
    for pid in range(num_pid_m * num_pid_n):
        yield grouped_tile(pid, num_pid_m, num_pid_n, group_m)


def distinct_tiles_loaded(tiles) -> Tuple[int, int]:
    """How many distinct A-tile rows and B-tile columns a set of programs needs."""
    rows = {m for m, _ in tiles}
    cols = {n for _, n in tiles}
    return len(rows), len(cols)
