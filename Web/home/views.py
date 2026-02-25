# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

"""
Django views for Scorecard-AI predictions.

This module is intentionally thin — all preprocessing lives in
``home.preprocessing`` and all input validation in ``home.schemas``.
"""

import logging

import numpy as np
import pandas as pd
from django.shortcuts import render

from .model_loader import get_models
from .preprocessing import build_feature_dataframe, get_feature_config, is_date_feature
from .schemas import validate_prediction_request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

def index(request):
    """Render the prediction form and handle submissions."""
    models = get_models()

    selected_feature_set = request.session.get("selected_feature_set", "F1")
    selected_features = request.session.get("selected_features", {})
    error_message = None
    prediction_result = None

    config = get_feature_config(selected_feature_set)

    if request.method == "POST":
        action = request.POST.get("action", "")

        if action == "select":
            # User changed the feature-set dropdown.
            feature_set = request.POST.get("feature_set", "F1")
            # Validate feature_set before storing in session
            if feature_set.upper() not in ['F1', 'F2']:
                feature_set = 'F1'  # Default to F1 if invalid
            request.session["selected_feature_set"] = feature_set
            selected_feature_set = feature_set
            config = get_feature_config(selected_feature_set)

        elif action == "submit":
            selected_model = request.POST.get("model", "")

            # ---- Step 1: Validate the request with Pydantic ----
            try:
                validated = validate_prediction_request(
                    feature_set=selected_feature_set,
                    model_name=selected_model,
                )
            except ValueError as exc:
                error_message = str(exc)
                return _render(
                    request, config, selected_feature_set,
                    selected_features, error_message, prediction_result,
                )

            # ---- Step 2: Collect raw input from the form ----
            selected_features_list = request.POST.getlist("selected_features")
            raw_input: dict[str, str] = {}
            for feature in selected_features_list:
                raw_val = request.POST.get(feature, "")
                if raw_val.strip():
                    raw_input[feature] = raw_val.strip()

            # ---- Step 3: Build validated DataFrame ----
            try:
                feature_df = build_feature_dataframe(raw_input, selected_feature_set)
            except ValueError as exc:
                error_message = str(exc)
                return _render(
                    request, config, selected_feature_set,
                    selected_features, error_message, prediction_result,
                )

            # ---- Step 4: Run prediction ----
            try:
                if selected_feature_set == "F1":
                    prediction_result = _predict_f1(feature_df, validated.model_name, models)
                else:
                    prediction_result = _predict_f2(feature_df, validated.model_name, models)
            except Exception as exc:
                logger.exception("Prediction failed")
                error_message = f"Prediction error: {exc}"

            # Persist selected features in the session.
            request.session["selected_features"] = {
                f: "" for f in selected_features_list
            }

    return _render(
        request, config, selected_feature_set,
        selected_features, error_message, prediction_result,
    )


# ---------------------------------------------------------------------------
# Template rendering helper
# ---------------------------------------------------------------------------

def _render(request, config, feature_set, selected_features, error, prediction):
    """Build the context dict and render the template."""
    return render(request, "home/index.html", {
        "models": config["models"],
        "all_features": config["features"],
        "date_features": config["date_features"],
        "selected_features": selected_features,
        "selected_feature_set": feature_set,
        "prediction_result": prediction,
        "error_message": error,
    })


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _predict_f1(features_df: pd.DataFrame, model_name: str, models: dict) -> str:
    """Run an F1 (regression) model and return a human-readable string."""
    model_map = {
        "Linear_regression": "f1_linear",
        "Decision_tree": "f1_decision_tree",
        "Random_forest": "f1_random_forest",
    }
    key = model_map.get(model_name)
    if not key or models.get(key) is None:
        return f"Model '{model_name}' is not available."

    value = models[key].predict(features_df)[0]
    if model_name == "Linear_regression" and value < 0:
        return "Negative prediction — try another model."
    return f"{value:.6f}"


def _predict_f2(features_df: pd.DataFrame, model_name: str, models: dict) -> str:
    """Run an F2 (classification) model and return a human-readable string."""
    model_map = {
        "Decision_tree": "f2_decision_tree",
        "Logistic": "f2_logistic",
        "Xgb": "f2_xgb",
    }
    key = model_map.get(model_name)
    if not key or models.get(key) is None:
        return f"Model '{model_name}' is not available."

    arr = np.array(features_df, dtype=float).reshape(1, -1)
    prediction = models[key].predict(arr)
    label = int(prediction[0])
    if label == 1:
        return "1 — Interest likely to be repaid"
    return "0 — Interest unlikely to be repaid"
