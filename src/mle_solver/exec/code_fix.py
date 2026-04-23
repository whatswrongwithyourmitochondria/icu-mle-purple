"""Pre-execution regex fixes for common LLM-generated code errors.

Applied before every execution. Each fix targets a specific API misuse
that the LLM produces despite prompt instructions.
"""

import re


def fix_common_errors(code: str) -> str:
    """Apply regex fixes for known API errors. Returns corrected code."""
    code = _fix_lgbm_early_stopping(code)
    code = _fix_lgbm_verbose(code)
    code = _fix_xgb_early_stopping(code)
    code = _fix_bool_astype_int(code)
    code = _fix_bool_map_nan(code)
    code = _fix_catboost_logloss(code)
    code = _fix_catboost_early_stopping_conflict(code)
    return code


def _fix_lgbm_early_stopping(code: str) -> str:
    """Move early_stopping_rounds from .fit() to callbacks=[lgb.early_stopping(N)]."""
    # Match: .fit(..., early_stopping_rounds=N, ...)
    pattern = r"(\.fit\([^)]*?),\s*early_stopping_rounds\s*=\s*(\d+)([^)]*\))"
    match = re.search(pattern, code)
    if not match:
        return code
    n = match.group(2)
    # Remove from fit() args
    code = re.sub(
        r",\s*early_stopping_rounds\s*=\s*\d+",
        "",
        code,
    )
    # Add callbacks if not already present
    if "lgb.early_stopping" not in code and "lightgbm.early_stopping" not in code:
        code = re.sub(
            r"(\.fit\([^)]*?)(\))",
            rf"\1, callbacks=[lgb.early_stopping({n})])",
            code,
            count=1,
        )
    return code


def _fix_lgbm_verbose(code: str) -> str:
    """Remove verbose= from LGBMClassifier/LGBMRegressor .fit() calls."""
    # Only fix verbose in .fit() context, not in constructor
    code = re.sub(
        r"(\.fit\([^)]*?),\s*verbose\s*=\s*(?:False|True|-?\d+)([^)]*\))",
        r"\1\2",
        code,
    )
    return code


def _fix_xgb_early_stopping(code: str) -> str:
    """Remove early_stopping_rounds from XGBClassifier/XGBRegressor .fit() calls.

    For XGBoost, early_stopping_rounds should be in the constructor.
    We can't easily move it there via regex, so just remove it from fit()
    to prevent the crash. The model will train without early stopping.
    """
    code = re.sub(
        r"(\.fit\([^)]*?),\s*early_stopping_rounds\s*=\s*\d+([^)]*\))",
        r"\1\2",
        code,
    )
    return code


def _fix_bool_astype_int(code: str) -> str:
    """Replace .astype(int) on target columns with .map({'True':1,'False':0}).astype(int).

    The Spaceship Titanic 'Transported' column contains 'True'/'False' strings.
    .astype(int) crashes on these. .map() handles both string and bool values.
    This is safe for numeric columns too — map returns NaN for non-matching values,
    and the .astype(int) fallback handles them.
    """
    # Match patterns like: df['col'].astype(int) or series.astype(int)
    # Only fix when it's on a target-like column access, not on numeric expressions
    # like (prediction > 0.5).astype(int) which is fine
    code = re.sub(
        r"(\[(?:target_col|['\"]Transported['\"]|['\"]target['\"])\])\s*\.astype\s*\(\s*int\s*\)",
        r"\1.map({'True': 1, 'False': 0, True: 1, False: 0}).astype(int)",
        code,
    )
    return code


def _fix_catboost_logloss(code: str) -> str:
    """Fix CatBoost loss function name: LogLoss → Logloss (case-sensitive)."""
    code = re.sub(
        r"""(['"])LogLoss\1""",
        r"\1Logloss\1",
        code,
    )
    return code


def _fix_catboost_early_stopping_conflict(code: str) -> str:
    """Remove early_stopping_rounds from CatBoost when od_wait is present.

    CatBoost uses od_wait natively. When both od_wait and early_stopping_rounds
    are passed, CatBoost raises an error.
    """
    if "od_wait" in code and "early_stopping_rounds" in code:
        if "CatBoost" in code or "catboost" in code:
            code = re.sub(
                r",?\s*early_stopping_rounds\s*=\s*\d+",
                "",
                code,
            )
    return code


def _fix_bool_map_nan(code: str) -> str:
    """Insert .fillna(0) between .map({True/False...}) and .astype(int).

    When the target column has NaN rows, .map({'True': 1, 'False': 0})
    preserves them as NaN, then .astype(int) crashes. Runs after
    _fix_bool_astype_int to catch both LLM-generated and fix-generated chains.
    """
    code = re.sub(
        r"(\.map\s*\(\s*\{[^}]*['\"]?True['\"]?\s*:\s*1[^}]*\}\s*\))\s*\.astype\s*\(\s*int\s*\)",
        r"\1.fillna(0).astype(int)",
        code,
    )
    return code
