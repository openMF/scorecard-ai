# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# AI-Assisted Development Notice:
# Portions of this file were developed with the assistance of GitHub Copilot.
# All AI-generated code was reviewed, tested, and validated by the contributor.

"""
Comprehensive tests for Scorecard-AI preprocessing, validation, and views.

Run with:
    python manage.py test home -v2
"""

import json
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from django.test import TestCase, RequestFactory

from home.preprocessing import (
    REFERENCE_DATE,
    TRAINING_STATS,
    build_feature_dataframe,
    date_to_days,
    get_feature_config,
    is_date_feature,
    safe_float,
)
from home.schemas import (
    F1PredictionRequest,
    F2PredictionRequest,
    PredictionRequest,
    validate_prediction_request,
)


# ===========================================================================
# Tests for home.preprocessing
# ===========================================================================


class TestReferenceDate(TestCase):
    """Verify the reference date matches the training pipeline."""

    def test_reference_date_value(self):
        self.assertEqual(str(REFERENCE_DATE.date()), "1924-02-17")

    def test_reference_date_is_timestamp(self):
        self.assertIsInstance(REFERENCE_DATE, pd.Timestamp)


class TestDateToDays(TestCase):
    """Tests for ``date_to_days()``."""

    def test_iso_date_string(self):
        """2024-02-17 is exactly 100 years = 36525 days after 1924-02-17."""
        result = date_to_days("2024-02-17")
        expected = (pd.to_datetime("2024-02-17") - REFERENCE_DATE).days
        self.assertEqual(result, float(expected))

    def test_numeric_string_passthrough(self):
        """Already-converted numeric string should pass through."""
        self.assertEqual(date_to_days("35000"), 35000.0)
        self.assertEqual(date_to_days("35000.5"), 35000.5)

    def test_int_passthrough(self):
        self.assertEqual(date_to_days(35000), 35000.0)

    def test_float_passthrough(self):
        self.assertEqual(date_to_days(35000.5), 35000.5)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            date_to_days("")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            date_to_days(None)

    def test_invalid_date_raises(self):
        with self.assertRaises(ValueError):
            date_to_days("not-a-date")

    def test_various_date_formats(self):
        """Should handle common date formats via pandas."""
        result1 = date_to_days("2023-01-15")
        result2 = date_to_days("Jan 15, 2023")
        self.assertEqual(result1, result2)

    def test_negative_passthrough(self):
        """Negative day-counts (before reference date) should work."""
        self.assertEqual(date_to_days("-100"), -100.0)


class TestSafeFloat(TestCase):
    """Tests for ``safe_float()``."""

    def test_valid_integer_string(self):
        self.assertEqual(safe_float("42", "amount"), 42.0)

    def test_valid_float_string(self):
        self.assertEqual(safe_float("3.14", "rate"), 3.14)

    def test_actual_int(self):
        self.assertEqual(safe_float(10, "count"), 10.0)

    def test_actual_float(self):
        self.assertEqual(safe_float(2.5, "ratio"), 2.5)

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError) as ctx:
            safe_float("", "amount")
        self.assertIn("amount", str(ctx.exception))

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            safe_float(None, "x")

    def test_non_numeric_string_raises(self):
        with self.assertRaises(ValueError) as ctx:
            safe_float("abc", "principal_amount")
        self.assertIn("principal_amount", str(ctx.exception))


class TestGetFeatureConfig(TestCase):
    """Tests for ``get_feature_config()``."""

    def test_f1_returns_dict(self):
        config = get_feature_config("F1")
        self.assertIn("features", config)
        self.assertIn("defaults", config)
        self.assertIn("date_features", config)
        self.assertIn("models", config)

    def test_f2_returns_dict(self):
        config = get_feature_config("F2")
        self.assertIn("features", config)
        self.assertEqual(config["target"], "interest_repaid_derived")

    def test_case_insensitive(self):
        config = get_feature_config("f1")
        self.assertEqual(config["target"], "nominal_interest_rate_per_period")

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_feature_config("F3")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_feature_config("")


class TestIsDateFeature(TestCase):
    """Tests for ``is_date_feature()``."""

    def test_activation_date_is_date(self):
        self.assertTrue(is_date_feature("activation_date", "F1"))

    def test_amount_is_not_date(self):
        self.assertFalse(is_date_feature("amount", "F1"))

    def test_created_date_in_f2(self):
        self.assertTrue(is_date_feature("created_date", "F2"))

    def test_created_date_not_in_f1(self):
        # F1 doesn't have created_date, so is_date_feature should return False
        self.assertFalse(is_date_feature("created_date", "F1"))
        self.assertFalse(is_date_feature("created_date", "F1"))


class TestBuildFeatureDataframe(TestCase):
    """Tests for ``build_feature_dataframe()``."""

    def test_empty_input_uses_defaults(self):
        """When no user input is provided, all defaults should be used."""
        df = build_feature_dataframe({}, "F1")
        self.assertEqual(df.shape[0], 1)
        config = get_feature_config("F1")
        self.assertEqual(df.shape[1], len(config["features"]))
        # Check a specific default
        self.assertAlmostEqual(
            df["principal_amount"].iloc[0],
            config["defaults"]["principal_amount"],
            places=4,
        )

    def test_partial_input_fills_missing(self):
        """User-supplied values should override defaults."""
        raw = {"principal_amount": "25000", "amount": "6000"}
        df = build_feature_dataframe(raw, "F1")
        self.assertEqual(df["principal_amount"].iloc[0], 25000.0)
        self.assertEqual(df["amount"].iloc[0], 6000.0)

    def test_date_conversion_in_input(self):
        """Date strings should be converted to day-counts."""
        raw = {"activation_date": "2024-06-15"}
        df = build_feature_dataframe(raw, "F1")
        expected = float((pd.to_datetime("2024-06-15") - REFERENCE_DATE).days)
        self.assertEqual(df["activation_date"].iloc[0], expected)

    def test_column_order_matches_config(self):
        """Column order must match the feature list in training_stats.json."""
        df = build_feature_dataframe({}, "F1")
        config = get_feature_config("F1")
        self.assertListEqual(list(df.columns), config["features"])

    def test_column_order_f2(self):
        df = build_feature_dataframe({}, "F2")
        config = get_feature_config("F2")
        self.assertListEqual(list(df.columns), config["features"])

    def test_invalid_numeric_raises(self):
        raw = {"principal_amount": "not_a_number"}
        with self.assertRaises(ValueError) as ctx:
            build_feature_dataframe(raw, "F1")
        self.assertIn("principal_amount", str(ctx.exception))

    def test_invalid_date_raises(self):
        raw = {"activation_date": "not_a_date"}
        with self.assertRaises(ValueError) as ctx:
            build_feature_dataframe(raw, "F1")
        self.assertIn("date", str(ctx.exception).lower())

    def test_all_dtypes_are_float(self):
        df = build_feature_dataframe({}, "F1")
        for col in df.columns:
            self.assertTrue(
                pd.api.types.is_float_dtype(df[col]),
                f"Column '{col}' is not float: {df[col].dtype}",
            )

    def test_unknown_features_ignored(self):
        """Extra keys not in the feature list should be silently ignored."""
        raw = {"unknown_field": "999", "principal_amount": "10000"}
        df = build_feature_dataframe(raw, "F1")
        self.assertNotIn("unknown_field", df.columns)
        self.assertEqual(df["principal_amount"].iloc[0], 10000.0)


class TestTrainingStatsIntegrity(TestCase):
    """Verify training_stats.json is internally consistent."""

    def test_all_features_have_defaults(self):
        for key in ("F1", "F2"):
            config = get_feature_config(key)
            for feat in config["features"]:
                self.assertIn(
                    feat, config["defaults"],
                    f"Feature '{feat}' in {key} has no default value.",
                )

    def test_date_features_subset_of_features(self):
        for key in ("F1", "F2"):
            config = get_feature_config(key)
            date_set = set(config["date_features"])
            all_set = set(config["features"])
            self.assertTrue(
                date_set.issubset(all_set),
                f"{key} date_features not a subset of features: {date_set - all_set}",
            )

    def test_numeric_features_subset_of_features(self):
        for key in ("F1", "F2"):
            config = get_feature_config(key)
            num_set = set(config["numeric_features"])
            all_set = set(config["features"])
            self.assertTrue(
                num_set.issubset(all_set),
                f"{key} numeric_features not a subset of features: {num_set - all_set}",
            )

    def test_no_overlap_date_numeric(self):
        """Date and numeric feature lists should be disjoint."""
        for key in ("F1", "F2"):
            config = get_feature_config(key)
            overlap = set(config["date_features"]) & set(config["numeric_features"])
            self.assertEqual(
                len(overlap), 0,
                f"{key} has features in both date and numeric lists: {overlap}",
            )

    def test_date_plus_numeric_covers_all(self):
        """Every feature should be in either date_features or numeric_features."""
        for key in ("F1", "F2"):
            config = get_feature_config(key)
            covered = set(config["date_features"]) | set(config["numeric_features"])
            all_feats = set(config["features"])
            self.assertEqual(
                covered, all_feats,
                f"{key} missing from date/numeric lists: {all_feats - covered}",
            )


# ===========================================================================
# Tests for home.schemas
# ===========================================================================


class TestPredictionRequestSchema(TestCase):
    """Tests for Pydantic validation schemas."""

    def test_valid_f1_request(self):
        req = validate_prediction_request("F1", "Decision_tree")
        self.assertIsInstance(req, F1PredictionRequest)
        self.assertEqual(req.feature_set, "F1")
        self.assertEqual(req.model_name, "Decision_tree")

    def test_valid_f2_request(self):
        req = validate_prediction_request("F2", "Xgb")
        self.assertIsInstance(req, F2PredictionRequest)

    def test_case_insensitive_feature_set(self):
        req = validate_prediction_request("f1", "Decision_tree")
        self.assertEqual(req.feature_set, "F1")

    def test_invalid_feature_set_raises(self):
        with self.assertRaises(ValueError):
            validate_prediction_request("F3", "Decision_tree")

    def test_empty_model_raises(self):
        with self.assertRaises(ValueError):
            validate_prediction_request("F1", "")

    def test_wrong_model_for_f1_raises(self):
        with self.assertRaises(ValueError):
            validate_prediction_request("F1", "Xgb")

    def test_wrong_model_for_f2_raises(self):
        with self.assertRaises(ValueError):
            validate_prediction_request("F2", "Linear_regression")

    def test_all_f1_models_valid(self):
        for model in ("Linear_regression", "Decision_tree", "Random_forest"):
            req = validate_prediction_request("F1", model)
            self.assertEqual(req.model_name, model)

    def test_all_f2_models_valid(self):
        for model in ("Decision_tree", "Logistic", "Xgb"):
            req = validate_prediction_request("F2", model)
            self.assertEqual(req.model_name, model)

    def test_whitespace_model_raises(self):
        with self.assertRaises(ValueError):
            validate_prediction_request("F1", "   ")


# ===========================================================================
# Tests for home.views (integration)
# ===========================================================================


class TestIndexView(TestCase):
    """Integration tests for the main view."""

    def test_get_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_get_contains_feature_checkboxes(self):
        response = self.client.get("/")
        self.assertContains(response, "activation_date")
        self.assertContains(response, "principal_amount")

    def test_get_contains_model_dropdown(self):
        response = self.client.get("/")
        self.assertContains(response, "Select a model")

    def test_select_f2_changes_features(self):
        response = self.client.post("/", {
            "action": "select",
            "feature_set": "F2",
        })
        self.assertEqual(response.status_code, 200)
        # F2-specific features
        self.assertContains(response, "validatedon_date")
        self.assertContains(response, "created_date")
        self.assertContains(response, "term_frequency")

    def test_submit_without_model_shows_error(self):
        response = self.client.post("/", {
            "action": "submit",
            "model": "",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "error-message")

    def test_submit_with_invalid_model_shows_error(self):
        response = self.client.post("/", {
            "action": "submit",
            "model": "NonexistentModel",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "error-message")

    @patch("home.views.get_models")
    def test_submit_f1_with_defaults(self, mock_get_models):
        """Submit F1 with defaults — mock the model to avoid loading pickles."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([12.5])
        mock_get_models.return_value = {
            "f1_linear": mock_model,
            "f1_decision_tree": mock_model,
            "f1_random_forest": mock_model,
            "f2_decision_tree": mock_model,
            "f2_logistic": mock_model,
            "f2_xgb": mock_model,
        }
        response = self.client.post("/", {
            "action": "submit",
            "model": "Decision_tree",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "prediction-result")
        self.assertContains(response, "12.500000")

    @patch("home.views.get_models")
    def test_submit_f2_classification(self, mock_get_models):
        """Submit F2 — verify classification result rendering."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_get_models.return_value = {
            "f1_linear": mock_model,
            "f1_decision_tree": mock_model,
            "f1_random_forest": mock_model,
            "f2_decision_tree": mock_model,
            "f2_logistic": mock_model,
            "f2_xgb": mock_model,
        }
        # First switch to F2.
        session = self.client.session
        session["selected_feature_set"] = "F2"
        session.save()

        response = self.client.post("/", {
            "action": "submit",
            "model": "Xgb",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Interest likely to be repaid")

    @patch("home.views.get_models")
    def test_submit_with_date_input(self, mock_get_models):
        """Date string should be converted and prediction should succeed."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([5.0])
        mock_get_models.return_value = {
            "f1_linear": mock_model,
            "f1_decision_tree": mock_model,
            "f1_random_forest": mock_model,
            "f2_decision_tree": mock_model,
            "f2_logistic": mock_model,
            "f2_xgb": mock_model,
        }
        response = self.client.post("/", {
            "action": "submit",
            "model": "Decision_tree",
            "selected_features": ["activation_date", "principal_amount"],
            "activation_date": "2024-06-15",
            "principal_amount": "25000",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "prediction-result")

    def test_submit_with_bad_numeric_shows_error(self):
        response = self.client.post("/", {
            "action": "submit",
            "model": "Decision_tree",
            "selected_features": ["principal_amount"],
            "principal_amount": "not_a_number",
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "error-message")
        self.assertContains(response, "principal_amount")

    def test_error_message_html_class(self):
        """Error messages should use the error-message CSS class."""
        response = self.client.post("/", {
            "action": "submit",
            "model": "",
        })
        content = response.content.decode()
        self.assertIn('class="error-message"', content)

    def test_date_features_in_context(self):
        """The template should receive date_features for input type hints."""
        response = self.client.get("/")
        self.assertIn("date_features", response.context)
