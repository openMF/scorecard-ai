import pickle
import os
import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Singleton class to manage ML model lifecycle.
    Ensures models are loaded once and reused across all requests.
    """
    _instance = None
    _models = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self, base_dir):
        """Load all models from disk into memory cache."""
        if self._models:
            logger.info("Models already loaded, skipping reload.")
            return

        model_paths = {
            'model_f1_1': 'Nominal_models/linear_regression_model_new.pkl',
            'model_f1_2': 'Nominal_models/decision_tree_regressor_model_new.pkl',
            'model_f1_3': 'Nominal_models/random_forest_regressor_model_new.pkl',
            'model_f2_1': 'Intrest_model/decision_tree_model.pkl',
            'model_f2_2': 'Intrest_model/logistic_model.pkl',
            'model_f2_3': 'Intrest_model/xgb_model.pkl',
        }

        for name, path in model_paths.items():
            full_path = os.path.join(base_dir, path)
            try:
                with open(full_path, 'rb') as f:
                    self._models[name] = pickle.load(f)
                logger.info(f"Successfully loaded model: {name}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {full_path}")
                self._models[name] = None
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                self._models[name] = None

    def get_model(self, name):
        """Retrieve a cached model by name."""
        return self._models.get(name)

    @property
    def is_loaded(self):
        return bool(self._models)


class HomeConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "home"

    def ready(self):
        """Initialize model registry at server startup."""
        from django.conf import settings
        registry = ModelRegistry.get_instance()
        if not registry.is_loaded:
            registry.load_models(settings.BASE_DIR)