"""Legal whole-component switches between two first-round assignments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


Edge = tuple[int, int]  # detection row, tracker row


@dataclass(frozen=True)
class DifferenceComponent:
    a0_edges: tuple[Edge, ...]
    a1_edges: tuple[Edge, ...]

    def signature(self, tracker_ids: tuple[int, ...]) -> str:
        def render(edges: tuple[Edge, ...]) -> str:
            return ";".join(f"d{d}t{tracker_ids[t]}" for d, t in edges)
        return f"0={render(self.a0_edges)}|1={render(self.a1_edges)}"


def _edges(matched: np.ndarray, n_det: int, n_trk: int) -> set[Edge]:
    values = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    edges = {(int(d), int(t)) for d, t in values}
    if len(edges) != len(values):
        raise ValueError("duplicate association edge")
    if any(not (0 <= d < n_det and 0 <= t < n_trk) for d, t in edges):
        raise ValueError("association edge outside input")
    if len({d for d, _ in edges}) != len(edges) or len({t for _, t in edges}) != len(edges):
        raise ValueError("association is not one-to-one")
    return edges


def components(a0: np.ndarray, a1: np.ndarray, n_det: int, n_trk: int) -> list[DifferenceComponent]:
    base, alternative = _edges(a0, n_det, n_trk), _edges(a1, n_det, n_trk)
    different = base ^ alternative
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for d, t in different:
        dn, tn = ("d", d), ("t", t)
        adjacency[dn].add(tn)
        adjacency[tn].add(dn)
    seen: set[tuple[str, int]] = set()
    result = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        nodes = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(adjacency[node] - nodes)
        seen.update(nodes)
        dets = {v for k, v in nodes if k == "d"}
        trks = {v for k, v in nodes if k == "t"}
        base_part = tuple(sorted((d, t) for d, t in base if d in dets or t in trks))
        alt_part = tuple(sorted((d, t) for d, t in alternative if d in dets or t in trks))
        if not base_part and not alt_part:
            raise AssertionError("empty difference component")
        result.append(DifferenceComponent(base_part, alt_part))
    return result


def switched_result(
    a0: np.ndarray, component: DifferenceComponent, n_det: int, n_trk: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = _edges(a0, n_det, n_trk)
    if not set(component.a0_edges) <= base:
        raise ValueError("component does not belong to A0")
    mixed = (base - set(component.a0_edges)) | set(component.a1_edges)
    _edges(np.asarray(sorted(mixed), dtype=np.int64).reshape(-1, 2), n_det, n_trk)
    matched = np.asarray(sorted(mixed), dtype=np.int64).reshape(-1, 2)
    unmatched_d = np.asarray(sorted(set(range(n_det)) - {d for d, _ in mixed}), dtype=np.int64)
    unmatched_t = np.asarray(sorted(set(range(n_trk)) - {t for _, t in mixed}), dtype=np.int64)
    return matched, unmatched_d, unmatched_t
