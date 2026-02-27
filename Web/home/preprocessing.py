# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

"""
Centralized preprocessing module for Scorecard-AI.

This module is the single source of truth for:
  - Feature definitions (which features belong to F1 / F2)
  - Date-to-numeric conversion (matching the training pipeline)
  - Default (mean) values for missing features
  - Input validation helpers

Both the Django views and unit tests import from here, ensuring that
inference preprocessing always matches what the models were trained on.

Reference
---------
Training scripts:  nominal_interest_rate.py, interest_repaid_derived.py
The training pipeline converts every date column to
  (date − reference_date).days
where reference_date is the minimum date across all date columns in the
original dataset (1924-02-17).
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load training statistics from JSON (single source of truth)
# ---------------------------------------------------------------------------
_STATS_PATH = os.path.join(os.path.dirname(__file__), "training_stats.json")

with open(_STATS_PATH, "r") as _f:
    TRAINING_STATS: dict = json.load(_f)

REFERENCE_DATE: pd.Timestamp = pd.to_datetime(TRAINING_STATS["reference_date"])

# Convenience accessors
F1_CONFIG: dict = TRAINING_STATS["feature_sets"]["F1"]
F2_CONFIG: dict = TRAINING_STATS["feature_sets"]["F2"]


def get_feature_config(feature_set: str) -> dict:
    """Return the config dict for a given feature set ('F1' or 'F2')."""
    key = feature_set.upper()
    if key not in TRAINING_STATS["feature_sets"]:
        raise ValueError(f"Unknown feature set: {feature_set!r}. Expected 'F1' or 'F2'.")
    return TRAINING_STATS["feature_sets"][key]


def is_date_feature(feature_name: str, feature_set: str) -> bool:
    """Check whether *feature_name* is a date column for the given set."""
    config = get_feature_config(feature_set)
    return feature_name in config["date_features"]


# ---------------------------------------------------------------------------
# Date conversion — matches training pipeline exactly
# ---------------------------------------------------------------------------

def date_to_days(value: Any) -> float:
    """Convert a date-like value to days since *REFERENCE_DATE*.

    Accepted formats
    ----------------
    - A numeric string or float  →  returned as-is (already converted).
    - An ISO-8601 date string    →  (date − 1924-02-17).days
    - ``None`` or empty string   →  raises ``ValueError``

    This mirrors ``process_date_columns()`` in the training scripts.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError("Date value cannot be empty.")

    # Already numeric — the user pasted a raw day-count.
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.replace(".", "", 1).lstrip("-").isdigit():
        return float(value)

    # Parse as a date string.
    try:
        date_obj = pd.to_datetime(value)
    except Exception as exc:
        raise ValueError(
            f"Cannot parse date: {value!r}. Use YYYY-MM-DD format or a numeric day-count."
        ) from exc

    return float((date_obj - REFERENCE_DATE).days)


def safe_float(value: Any, feature_name: str) -> float:
    """Cast *value* to float with a clear error on failure."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValueError(f"Value for '{feature_name}' cannot be empty.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot convert '{feature_name}' value {value!r} to a number."
        ) from exc


# ---------------------------------------------------------------------------
# Build a model-ready DataFrame from raw user input
# ---------------------------------------------------------------------------

def build_feature_dataframe(
    raw_input: dict[str, Any],
    feature_set: str,
) -> pd.DataFrame:
    """Construct a single-row DataFrame ready for model prediction.

    Parameters
    ----------
    raw_input:
        Mapping of ``feature_name → raw_value`` for every feature the
        user chose to fill in.  Missing keys are filled with the
        training-set mean from *training_stats.json*.
    feature_set:
        ``'F1'`` (nominal interest rate) or ``'F2'`` (interest repaid).

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns in the exact order the model
        expects, all values cast to ``float``.

    Raises
    ------
    ValueError
        If any value cannot be parsed or converted.
    """
    config = get_feature_config(feature_set)
    defaults = config["defaults"]
    date_cols = set(config["date_features"])
    errors: list[str] = []
    row: dict[str, float] = {}

    for feature in config["features"]:
        if feature in raw_input and raw_input[feature] not in (None, ""):
            raw_val = raw_input[feature]
            try:
                if feature in date_cols:
                    row[feature] = date_to_days(raw_val)
                else:
                    row[feature] = safe_float(raw_val, feature)
            except ValueError as exc:
                errors.append(str(exc))
        else:
            # Use training-set mean as fallback
            row[feature] = defaults[feature]

    if errors:
        raise ValueError(" | ".join(errors))

    return pd.DataFrame([row], columns=config["features"])
