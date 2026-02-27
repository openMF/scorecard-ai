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
import json

import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

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
# API v1
# ---------------------------------------------------------------------------


@csrf_exempt
@require_POST
def predict_api_v1(request):
    """JSON API endpoint for F1/F2 prediction requests."""
    models = get_models()

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return _api_error(
            message="Invalid JSON payload.",
            code="INVALID_JSON",
            status=400,
        )

    feature_set = (payload.get("feature_set") or "").strip().upper()
    model_name = (payload.get("model_name") or payload.get("model") or "").strip()
    if "inputs" in payload:
        raw_input = payload.get("inputs")
    elif "features" in payload:
        raw_input = payload.get("features")
    else:
        raw_input = {}

    if not isinstance(raw_input, dict):
        return _api_error(
            message="'inputs' must be an object of feature_name -> value.",
            code="INVALID_INPUTS",
            status=400,
        )

    try:
        validated = validate_prediction_request(
            feature_set=feature_set,
            model_name=model_name,
        )
        feature_df = build_feature_dataframe(raw_input, validated.feature_set)
        result = _predict_api_payload(feature_df, validated.feature_set, validated.model_name, models)
    except ValueError as exc:
        return _api_error(
            message=str(exc),
            code="VALIDATION_ERROR",
            status=422,
        )
    except Exception as exc:
        logger.exception("API prediction failed")
        return _api_error(
            message=f"Prediction error: {exc}",
            code="PREDICTION_ERROR",
            status=500,
        )

    response = {
        "success": True,
        "data": {
            "feature_set": validated.feature_set,
            "task": "regression" if validated.feature_set == "F1" else "classification",
            "model_name": validated.model_name,
            "prediction": result["prediction"],
            "prediction_display": result["prediction_display"],
            "confidence": result.get("confidence"),
            "normalized_inputs": feature_df.iloc[0].to_dict(),
        },
    }
    return JsonResponse(response, status=200)


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


def _api_error(message: str, code: str, status: int = 400):
    """Return a standardized API error payload."""
    return JsonResponse(
        {
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
        status=status,
    )


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


def _predict_api_payload(
    features_df: pd.DataFrame,
    feature_set: str,
    model_name: str,
    models: dict,
) -> dict:
    """Run prediction and return structured response payload fields."""
    if feature_set == "F1":
        model_map = {
            "Linear_regression": "f1_linear",
            "Decision_tree": "f1_decision_tree",
            "Random_forest": "f1_random_forest",
        }
        key = model_map.get(model_name)
        if not key or models.get(key) is None:
            raise ValueError(f"Model '{model_name}' is not available.")

        raw_value = float(models[key].predict(features_df)[0])
        if model_name == "Linear_regression" and raw_value < 0:
            display = "Negative prediction — try another model."
        else:
            display = f"{raw_value:.6f}"

        return {
            "prediction": raw_value,
            "prediction_display": display,
        }

    model_map = {
        "Decision_tree": "f2_decision_tree",
        "Logistic": "f2_logistic",
        "Xgb": "f2_xgb",
    }
    key = model_map.get(model_name)
    if not key or models.get(key) is None:
        raise ValueError(f"Model '{model_name}' is not available.")

    arr = np.array(features_df, dtype=float).reshape(1, -1)
    model = models[key]
    prediction = model.predict(arr)
    label = int(prediction[0])
    display = "1 — Interest likely to be repaid" if label == 1 else "0 — Interest unlikely to be repaid"

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(arr)
            confidence = float(np.max(probs[0]))
        except Exception:
            confidence = None

    return {
        "prediction": label,
        "prediction_display": display,
        "confidence": confidence,
    }
