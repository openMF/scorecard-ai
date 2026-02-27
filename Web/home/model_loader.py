# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

"""
Singleton model loader — loads all ML models once and caches them in memory.
This avoids re-reading 384 MB of pickle files on every HTTP request.
"""
import pickle
import os
import warnings
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

_models_cache = {}


def _load_all_models():
    """Load all models from disk into the cache dict."""
    base = settings.BASE_DIR

    model_files = {
        "f1_linear": os.path.join(base, "Nominal_models", "linear_regression_model_new.pkl"),
        "f1_decision_tree": os.path.join(base, "Nominal_models", "decision_tree_regressor_model_new.pkl"),
        "f1_random_forest": os.path.join(base, "Nominal_models", "random_forest_regressor_model_new.pkl"),
        "f2_decision_tree": os.path.join(base, "Intrest_model", "decision_tree_model.pkl"),
        "f2_logistic": os.path.join(base, "Intrest_model", "logistic_model.pkl"),
        "f2_xgb": os.path.join(base, "Intrest_model", "xgb_model.pkl"),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, path in model_files.items():
            try:
                with open(path, "rb") as f:
                    _models_cache[key] = pickle.load(f)
                logger.info("Loaded model %s from %s", key, path)
            except Exception as e:
                logger.error("Failed to load model %s from %s: %s", key, path, e)
                _models_cache[key] = None


def get_models():
    """Return the cached models dict, loading them on first call."""
    if not _models_cache:
        _load_all_models()
    return _models_cache
