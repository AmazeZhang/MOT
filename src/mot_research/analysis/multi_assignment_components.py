"""Whole connected components over two or more legal one-to-one assignments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np


Edge = tuple[int, int]


def checked_edges(matched: np.ndarray, n_det: int, n_trk: int) -> frozenset[Edge]:
    values = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    edges = frozenset((int(d), int(t)) for d, t in values)
    if len(edges) != len(values):
        raise ValueError("duplicate association edge")
    if any(not (0 <= d < n_det and 0 <= t < n_trk) for d, t in edges):
        raise ValueError("association edge outside input")
    if len({d for d, _ in edges}) != len(edges) or len({t for _, t in edges}) != len(edges):
        raise ValueError("association is not one-to-one")
    return edges


@dataclass(frozen=True)
class MultiAssignmentComponent:
    detection_rows: tuple[int, ...]
    tracker_rows: tuple[int, ...]
    actions: tuple[tuple[str, tuple[Edge, ...]], ...]

    def action(self, name: str) -> tuple[Edge, ...]:
        return dict(self.actions)[name]

    def unique_actions(self, order: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[tuple[Edge, ...]] = set()
        result = []
        available = dict(self.actions)
        for name in order:
            edges = available[name]
            if edges not in seen:
                result.append(name)
                seen.add(edges)
        return tuple(result)


def components(assignments: Mapping[str, np.ndarray], n_det: int, n_trk: int,
               *, reference: str) -> list[MultiAssignmentComponent]:
    if reference not in assignments or len(assignments) < 2:
        raise ValueError("reference and at least two assignments are required")
    edge_sets = {name: checked_edges(value, n_det, n_trk)
                 for name, value in assignments.items()}
    union = frozenset().union(*edge_sets.values())
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for d, t in union:
        dn, tn = ("d", d), ("t", t)
        adjacency[dn].add(tn)
        adjacency[tn].add(dn)
    result = []
    seen: set[tuple[str, int]] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        nodes: set[tuple[str, int]] = set()
        while stack:
            node = stack.pop()
            if node in nodes:
                continue
            nodes.add(node)
            stack.extend(adjacency[node] - nodes)
        seen.update(nodes)
        dets = tuple(sorted(value for kind, value in nodes if kind == "d"))
        trks = tuple(sorted(value for kind, value in nodes if kind == "t"))
        actions = tuple((name, tuple(sorted((d, t) for d, t in edges
                                             if d in dets or t in trks)))
                        for name, edges in assignments_order(edge_sets, reference))
        if len({edges for _, edges in actions}) == 1:
            continue
        result.append(MultiAssignmentComponent(dets, trks, actions))
    return result


def assignments_order(edge_sets: Mapping[str, frozenset[Edge]], reference: str):
    yield reference, edge_sets[reference]
    for name in edge_sets:
        if name != reference:
            yield name, edge_sets[name]


def apply_action(reference_assignment: np.ndarray, component: MultiAssignmentComponent,
                 action: str, n_det: int, n_trk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = checked_edges(reference_assignment, n_det, n_trk)
    reference_part = set(component.actions[0][1])
    # Components are constructed with the reference action first.
    if not reference_part <= reference:
        raise ValueError("component reference projection is not in reference assignment")
    mixed = (set(reference) - reference_part) | set(component.action(action))
    matched = np.asarray(sorted(mixed), dtype=np.int64).reshape(-1, 2)
    checked_edges(matched, n_det, n_trk)
    unmatched_d = np.asarray(sorted(set(range(n_det)) - {d for d, _ in mixed}), dtype=np.int64)
    unmatched_t = np.asarray(sorted(set(range(n_trk)) - {t for _, t in mixed}), dtype=np.int64)
    return matched, unmatched_d, unmatched_t
