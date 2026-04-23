"""Top-level entry point.

Usage:
    from mle_solver.runner import run_competition
    submission_bytes = run_competition(work_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config
from .ensemble import blend_submissions
from .llm import LLMClient
from .panel import PanelResult, run_panel
from .protocol import SPLIT_CSV, PROTOCOL_JSON, infer_contract, prepare_splits
from .tree.loop import RunContext
from .prompts import disposition_for_run

logger = logging.getLogger("mle-solver")


_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "mle-solver.yaml"


def run_competition(work_dir: Path) -> bytes:
    """Run the full panel against a competition workspace.

    ``work_dir`` must contain ``home/data/`` with ``description.md``,
    ``train.csv`` (or equivalent), and ``sample_submission.csv``.
    """
    work_dir = Path(work_dir)
    data_dir = work_dir / "home" / "data"
    workspace_root = work_dir / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    _configure_logging(cfg)

    cfg_errs = cfg.validate()
    if cfg_errs:
        raise RuntimeError(f"[runner] config invalid: {'; '.join(cfg_errs)}")
    if not data_dir.exists():
        raise FileNotFoundError(f"[runner] data dir missing: {data_dir}")

    # ── Runner-owned protocol ─────────────────────────────────────────────
    llm = LLMClient(cfg.llm)
    contract = infer_contract(
        data_dir,
        llm=llm,
        n_folds=cfg.search.n_folds,
        holdout_fraction=cfg.search.holdout_fraction,
        seed=cfg.seed,
    )
    splits = prepare_splits(data_dir, contract)
    if splits is None:
        logger.warning(
            "[runner] could not prepare splits — solver will need to carve its own"
        )
    contract_summary = _render_contract_summary(contract, splits)
    task_desc = _read_description(data_dir)
    data_files = _list_data_files(data_dir)
    data_preview = _build_data_preview(data_dir)
    env_summary = _env_summary()
    direction_label = "higher is better" if contract.maximize else "lower is better"
    sample_submission_path = data_dir / "sample_submission.csv"

    def build_context(seat_cfg: Config, seat_index: int) -> RunContext:
        return RunContext(
            task_desc=task_desc,
            data_files=data_files,
            data_preview=data_preview,
            env_summary=env_summary,
            contract_summary=contract_summary,
            maximize=contract.maximize,
            direction_label=direction_label,
            sample_submission_path=sample_submission_path,
            disposition=disposition_for_run(seat_index, cfg.search.dispositions or None),
        )

    result: PanelResult = run_panel(
        cfg=cfg,
        data_dir=data_dir,
        workspace_root=workspace_root,
        build_context=build_context,
    )

    if not result.final_candidates:
        raise RuntimeError("[runner] panel produced no candidates")

    # Try blending only clean/non-leaky candidates
    candidates = result.final_candidates
    clean = [n for n in candidates if n.review_verdict not in ("leaky",) and n.submission_path and n.submission_path.exists()]
    if not clean:
        clean = [n for n in candidates if n.submission_path and n.submission_path.exists()]
    paths = [n.submission_path for n in clean]
    holdouts = [n.holdout_score for n in clean]

    if len(paths) >= 2:
        blended = blend_submissions(paths, holdout_scores=holdouts, maximize=contract.maximize)
        if blended is not None:
            logger.info(f"[runner] shipping blended submission from {len(paths)} candidates")
            return blended

    best = candidates[0]
    sub_path = best.submission_path
    if sub_path is None or not sub_path.exists():
        raise RuntimeError(f"[runner] best candidate has no submission file: {sub_path}")
    logger.info(f"[runner] shipping best: {best.short()}")
    return sub_path.read_bytes()


# ── helpers ──────────────────────────────────────────────────────────────


def _load_config() -> Config:
    cfg = Config.from_yaml(_CONFIG_PATH)
    cfg.resolve_env()
    return cfg


def _configure_logging(cfg: Config) -> None:
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
        logger.addHandler(handler)


def _render_contract_summary(contract, splits) -> str:
    lines = [
        f"- metric: {contract.metric} ({'maximize' if contract.maximize else 'minimize'})",
        f"- target_col: {contract.target_col or '(inspect sample_submission)'}",
        f"- id_col: {contract.id_col}",
        f"- category: {contract.category}",
        f"- n_folds: {contract.n_folds}",
    ]
    if splits is not None:
        lines.append(
            f"- split file: ./input/{SPLIT_CSV} "
            f"(n_dev={splits.n_dev}, n_holdout={splits.n_holdout})"
        )
        lines.append(f"- protocol file: ./input/{PROTOCOL_JSON}")
    else:
        lines.append("- split file: MISSING — carve your own dev/holdout using the protocol above")
    return "\n".join(lines)


def _read_description(data_dir: Path) -> str:
    return (data_dir / "description.md").read_text(encoding="utf-8")


def _list_data_files(data_dir: Path) -> list[str]:
    out: list[str] = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and not p.name.startswith("_"):
            rel = p.relative_to(data_dir)
            out.append(str(rel))
    return out


def _build_data_preview(data_dir: Path, max_chars: int = 2000) -> str:
    import pandas as pd
    parts: list[str] = []
    for csv in sorted(data_dir.glob("*.csv")):
        if csv.name.startswith("_"):
            continue
        df = pd.read_csv(csv, nrows=3)
        parts.append(f"# {csv.name}\n{df.to_string()}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n\n".join(parts)[:max_chars]


def _env_summary() -> str:
    import platform
    return f"- Python {platform.python_version()} on {platform.system()}"


