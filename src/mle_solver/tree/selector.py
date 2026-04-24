"""UCB-over-branches selector with debug-first logic.

A "branch" is the lineage rooted at a single draft node. At each step we:

1. Pick a debug target if any buggy node still has spare budget (debug-first).
2. Otherwise, pick the branch with the highest UCB score and improve its best
   current valid descendant.

UCB score per branch:
    mean_reward + c * sqrt( ln(total_plays) / branch_plays )

mean_reward is the best valid holdout_score in the branch (excluding nodes
flagged leaky with medium/high confidence), normalized so the best branch
is 1 and the worst is 0 (ties = 1). branch_plays is the number of improve
nodes rooted in that branch so far.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .journal import Journal
from .node import SearchNode

logger = logging.getLogger("mle-solver")


@dataclass
class NextAction:
    kind: str              # "debug" | "improve"
    parent: SearchNode


MAX_IMPROVE_ATTEMPTS_PER_NODE = 10
MAX_DEBUG_CHAIN_DEPTH = 5
BRANCH_STAGNATION_LIMIT = 15


class Selector:
    def __init__(self, *, max_debug_attempts_per_node: int, explore_c: float = 1.0):
        self.max_debug_attempts = int(max_debug_attempts_per_node)
        self.c = float(explore_c)

    def pick(self, journal: Journal, *, excluded: set[str] | None = None, maximize: bool = True) -> NextAction | None:
        excluded = excluded or set()

        debug_target = self._pick_debug(journal, excluded)
        if debug_target is not None:
            debug_target.debug_attempts += 1
            logger.info(
                f"[selector] debug pick: {debug_target.id} "
                f"(attempts={debug_target.debug_attempts}/{self.max_debug_attempts})"
            )
            return NextAction(kind="debug", parent=debug_target)

        improve_target = self._pick_improve_ucb(journal, excluded, maximize)
        if improve_target is not None:
            return NextAction(kind="improve", parent=improve_target)
        return None

    # ── debug ────────────────────────────────────────────────────────────

    def _pick_debug(self, journal: Journal, excluded: set[str]) -> SearchNode | None:
        for node in reversed(list(journal)):
            if not node.is_buggy:
                continue
            if node.id in excluded:
                continue
            if node.debug_attempts >= self.max_debug_attempts:
                continue
            root = node.branch_root_id or node.id
            if self._branch_has_valid(journal, root):
                continue
            if self._branch_debug_count(journal, root) >= MAX_DEBUG_CHAIN_DEPTH:
                continue
            return node
        return None

    @staticmethod
    def _branch_has_valid(journal: Journal, root: str) -> bool:
        for n in journal:
            if (n.branch_root_id or n.id) == root and n.is_valid:
                return True

    @staticmethod
    def _branch_debug_count(journal: Journal, root: str) -> int:
        return sum(1 for n in journal if (n.branch_root_id or n.id) == root and n.stage == "debug")
        return False

    # ── improve (UCB) ────────────────────────────────────────────────────

    def _pick_improve_ucb(
        self,
        journal: Journal,
        excluded: set[str],
        maximize: bool,
    ) -> SearchNode | None:
        branches = journal.branches()
        candidates = self._ucb_candidates(branches, excluded, maximize, filter_leaky=True)
        if not candidates:
            return None

        # Normalize holdout scores to [0, 1] across branches for the exploitation term.
        scores = [c[3] for c in candidates]
        lo, hi = min(scores), max(scores)
        spread = hi - lo if hi > lo else 1.0
        normed = [(s - lo) / spread for s in scores]

        total_plays = sum(c[2] for c in candidates) or 1
        best_idx = -1
        best_ucb = float("-inf")
        for i, (root_id, node, plays, _raw) in enumerate(candidates):
            mean_reward = normed[i]
            explore = self.c * math.sqrt(math.log(total_plays + 1) / (plays + 1))
            ucb = mean_reward + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        root_id, chosen, plays, _ = candidates[best_idx]
        logger.info(
            f"[selector] improve pick (UCB): node={chosen.id} branch={root_id} "
            f"plays={plays} ucb={best_ucb:.3f}"
        )
        return chosen


    @staticmethod
    def _ucb_candidates(
        branches: dict[str, list[SearchNode]],
        excluded: set[str],
        maximize: bool,
        filter_leaky: bool,
    ) -> list[tuple[str, SearchNode, int, float]]:
        # Count how many times each node has been an improve parent.
        improve_parent_counts: dict[str, int] = {}
        for nodes in branches.values():
            for n in nodes:
                if n.stage == "improve" and n.parent_id:
                    improve_parent_counts[n.parent_id] = improve_parent_counts.get(n.parent_id, 0) + 1

        candidates: list[tuple[str, SearchNode, int, float]] = []
        for root_id, nodes in branches.items():
            valid = [
                n for n in nodes
                if n.is_valid
                and n.id not in excluded
                and (not filter_leaky or n.review_verdict != "leaky")
                and improve_parent_counts.get(n.id, 0) < MAX_IMPROVE_ATTEMPTS_PER_NODE
            ]
            if not valid:
                continue
            best = max(
                valid,
                key=lambda n: (n.holdout_score if maximize else -n.holdout_score) if n.holdout_score is not None else float("-inf"),
            )
            plays = sum(1 for n in nodes if n.stage == "improve")
            if _is_stagnant(nodes, best, plays, maximize):
                continue
            ho = best.holdout_score if best.holdout_score is not None else 0.0
            candidates.append((root_id, best, plays, ho if maximize else -ho))
        return candidates


def _is_stagnant(
    nodes: list[SearchNode], best: SearchNode, plays: int, maximize: bool,
) -> bool:
    if plays < BRANCH_STAGNATION_LIMIT:
        return False
    best_ho = best.holdout_score
    if best_ho is None:
        return False
    improve_nodes = sorted(
        [n for n in nodes if n.stage == "improve" and n.is_valid],
        key=lambda n: n.created_at,
    )
    recent = improve_nodes[-BRANCH_STAGNATION_LIMIT:]
    for n in recent:
        ho = n.holdout_score
        if ho is not None:
            better = ho > best_ho if maximize else ho < best_ho
            if better:
                return False
    return True

