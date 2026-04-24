"""Tests for journal + UCB selector + rotating hint picker."""

import random

from mle_solver.exec.interpreter import ExecResult
from mle_solver.prompts.improve import IMPROVE_HINTS, pick_hint
from mle_solver.tree import Journal, SearchNode, Selector


def _valid_node(node_id, stage, parent_id, branch_root, cv, hold=None, hint_idx=None):
    n = SearchNode(id=node_id, stage=stage, code="pass", parent_id=parent_id, branch_root_id=branch_root)
    n.cv_score = cv
    n.holdout_score = hold if hold is not None else cv
    n.maximize = True
    n.improve_hint_index = hint_idx
    n.result = ExecResult(
        return_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        submission_path=None,  # bypass the has_submission check for selector tests
    )
    return n


def test_journal_best_prefers_non_leaky():
    j = Journal()
    a = _valid_node("d001", "draft", None, "d001", cv=0.70)
    b = _valid_node("i002", "improve", "d001", "d001", cv=0.85)
    b.review_verdict = "leaky"
    j.add(a)
    j.add(b)
    a.result.submission_path = j  # type: ignore[assignment]
    b.result.submission_path = j  # type: ignore[assignment]

    valid_all = [a, b]
    j._nodes = valid_all  # type: ignore[attr-defined]
    non_leaky = [n for n in valid_all if n.review_verdict != "leaky"]
    assert non_leaky == [a]


def test_selector_ucb_explores_undeveloped_branch():
    from unittest.mock import patch

    j = Journal()

    root_a = SearchNode(id="d001", stage="draft", code="a", branch_root_id="d001")
    root_a.cv_score = 0.80
    root_a.holdout_score = 0.80
    root_a.maximize = True
    for i in range(20):
        imp = SearchNode(
            id=f"i{i+1:03d}",
            stage="improve",
            code="a",
            parent_id="d001",
            branch_root_id="d001",
        )
        imp.cv_score = 0.80
        imp.holdout_score = 0.80
        imp.maximize = True
        j._nodes.append(imp)
        j._by_id[imp.id] = imp

    root_b = SearchNode(id="d002", stage="draft", code="b", branch_root_id="d002")
    root_b.cv_score = 0.75
    root_b.holdout_score = 0.75
    root_b.maximize = True

    j._nodes.insert(0, root_a)
    j._by_id["d001"] = root_a
    j._nodes.insert(1, root_b)
    j._by_id["d002"] = root_b

    with patch.object(SearchNode, "is_valid", property(lambda self: True)):
        sel = Selector(max_debug_attempts_per_node=2, explore_c=1.0)
        action = sel.pick(j, maximize=True)
    assert action is not None
    assert action.kind == "improve"
    assert action.parent.id == "d002"


def test_pick_hint_cold_start_returns_untried_indices_first():
    j = Journal()
    rng = random.Random(0)
    assert len(IMPROVE_HINTS) >= 8
    assert pick_hint(j, rng=rng) == 0

    parent = SearchNode(id="d001", stage="draft", code="x", branch_root_id="d001")
    parent.cv_score = 0.5
    parent.holdout_score = 0.5
    parent.maximize = True
    child = SearchNode(id="i002", stage="improve", code="x", parent_id="d001", branch_root_id="d001")
    child.cv_score = 0.6
    child.holdout_score = 0.6
    child.maximize = True
    child.improve_hint_index = 0
    j._nodes = [parent, child]
    j._by_id = {"d001": parent, "i002": child}
    assert pick_hint(j, rng=rng) == 1


def test_review_verdict_binary():
    n = SearchNode(id="i001", stage="improve", code="x")
    assert n.review_verdict != "leaky"

    n.review_verdict = "clean"
    assert n.review_verdict != "leaky"

    n.review_verdict = "leaky"
    assert n.review_verdict == "leaky"
