# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

"""
Pydantic schemas for validating user input before it reaches the ML models.

These schemas enforce type constraints and value ranges *before* the data
is assembled into a DataFrame, giving clear error messages instead of
opaque NumPy / sklearn crashes.
"""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared field constraints (derived from training-data statistics)
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Base schema shared by both F1 and F2 prediction requests."""

    feature_set: str = Field(
        ...,
        description="'F1' (nominal interest rate) or 'F2' (interest repaid).",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the ML model to use for prediction.",
    )

    @field_validator("feature_set")
    @classmethod
    def validate_feature_set(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in ("F1", "F2"):
            raise ValueError("feature_set must be 'F1' or 'F2'.")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Please select a model before submitting.")
        return v.strip()


class F1PredictionRequest(PredictionRequest):
    """Validation schema for F1 — Nominal Interest Rate prediction."""

    VALID_MODELS: ClassVar[set[str]] = {"Linear_regression", "Decision_tree", "Random_forest"}

    @field_validator("model_name")
    @classmethod
    def validate_f1_model(cls, v: str) -> str:
        v = v.strip()
        if v not in cls.VALID_MODELS:
            raise ValueError(
                f"Invalid F1 model: '{v}'. "
                f"Choose from: {', '.join(sorted(cls.VALID_MODELS))}."
            )
        return v


class F2PredictionRequest(PredictionRequest):
    """Validation schema for F2 — Interest Repaid prediction."""

    VALID_MODELS: ClassVar[set[str]] = {"Decision_tree", "Logistic", "Xgb"}

    @field_validator("model_name")
    @classmethod
    def validate_f2_model(cls, v: str) -> str:
        v = v.strip()
        if v not in cls.VALID_MODELS:
            raise ValueError(
                f"Invalid F2 model: '{v}'. "
                f"Choose from: {', '.join(sorted(cls.VALID_MODELS))}."
            )
        return v


class FeatureValue(BaseModel):
    """Schema for a single feature value submitted by the user."""

    name: str = Field(..., min_length=1)
    value: str = Field(..., description="Raw string value from the form.")
    is_date: bool = Field(default=False)

    @field_validator("value")
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Feature value cannot be empty.")
        return v.strip()


def validate_prediction_request(
    feature_set: str,
    model_name: str,
) -> PredictionRequest:
    """Factory that returns the correct validated schema instance.

    Raises ``ValueError`` with a user-friendly message if validation fails.
    """
    feature_set = feature_set.strip().upper() if feature_set else ""

    try:
        if feature_set == "F1":
            return F1PredictionRequest(
                feature_set=feature_set,
                model_name=model_name or "",
            )
        elif feature_set == "F2":
            return F2PredictionRequest(
                feature_set=feature_set,
                model_name=model_name or "",
            )
        else:
            return PredictionRequest(
                feature_set=feature_set or "INVALID",
                model_name=model_name or "",
            )
    except Exception as exc:
        # Flatten Pydantic errors into a readable string.
        if hasattr(exc, "errors"):
            messages = [e["msg"] for e in exc.errors()]
            raise ValueError(" | ".join(messages)) from exc
        raise
