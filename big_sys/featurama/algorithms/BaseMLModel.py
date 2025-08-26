# Model building algorithms

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import io
import json
import base64
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
from django.db import models


class MachineLearningError(Exception):
    """Custom exception for machine learning errors."""
    pass


class BaseMachineLearning(ABC):
    def __init__(self, pipeline: models.Model):
        self.pipeline = pipeline
        self._validate_input()
        self.params = self.pipeline.ml_model_parameters

    def _validate_input(self) -> None:
        if not self.pipeline.data_content:
            raise MachineLearningError("Pipeline has no dataset attached")

        if not self.pipeline.target_variable:
            raise MachineLearningError("Pipeline has no target variable")

        if not self.pipeline.final_selected_features:
            raise MachineLearningError("Empty final selected feature list")

        if not self.pipeline.ml_model:
            raise MachineLearningError("Wasn't choosen model method")

        if not self.pipeline.ml_model_parameters:
            raise MachineLearningError("Empty model method's params")

    def _get_features_data(self) -> pd.DataFrame:
        """Get DataFrame with only final selected features"""
        df = self.pipeline.get_dataframe()
        return df[self.pipeline.final_selected_features]

    def _get_target_data(self) -> pd.Series:
        """Get target Series"""
        df = self.pipeline.get_dataframe()
        return df[self.pipeline.target_variable]

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X_scaled

    def _generate_global_shap_plot(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray
    ) -> bytes:
        """Generate a global SHAP plot."""
        plt.figure(figsize=(14, 6), dpi=150)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=X.columns,
            plot_type="bar",
            show=False
        )
        plt.xlabel(
            "mean(|SHAP value|) (impact on model output magnitude)",
            fontsize=12
        )
        plt.tick_params(axis='y', labelsize=10)
        plt.title("Информативность признаков", fontsize=14)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    def _generate_distribution_shap_plot(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray
    ) -> bytes:
        """Generate a SHAP-distribution plot."""
        plt.figure(figsize=(14, 8), dpi=150)
        shap.summary_plot(
            shap_values,
            X,
            feature_names=X.columns,
            show=False
        )
        plt.xlabel(
            "SHAP value (impact on model output)",
            fontsize=12
        )
        plt.tick_params(axis='y', labelsize=10)
        plt.title(
            "Распределение SHAP-значений по признакам",
            fontsize=14
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()

    def _save_result(self, roc_auc, accuracy,
                     f1_score, precision, recall,
                     explainer, shap_values, 
                     global_shap_plot, distribution_shap_plot) -> None:
        """Save results"""
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        }

        shap_results = {
            # 'explainer': explainer,
            'shap_values': shap_values.tolist(),
            'global_shap_plot': base64.b64encode(global_shap_plot).decode('utf-8'),
            'distribution_shap_plot': base64.b64encode(distribution_shap_plot).decode('utf-8'),
        }
        self.pipeline.ml_model_metrics = json.dumps(metrics)
        self.pipeline.ml_model_shap_values = json.dumps(shap_results)
        self.pipeline.save()

    # SHAP Methods
    def shap_linear_explainer(self, model: Any,
                              X: pd.DataFrame, **kwargs) -> Tuple[shap.Explainer, np.ndarray]:
        """Compute SHAP values using LinearExplainer."""
        explainer = shap.LinearExplainer(model, X, **kwargs)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values

    def shap_tree_explainer(self, model: Any,
                            X: pd.DataFrame, **kwargs) -> Tuple[shap.Explainer, np.ndarray]:
        """Compute SHAP values using TreeExplainer."""
        explainer = shap.TreeExplainer(model, **kwargs)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, np.ndarray):
            if len(shap_values.shape) == 2:
                return explainer, shap_values
            elif len(shap_values.shape) == 3:
                return explainer, shap_values[:, :, 1]
            else:
                raise MachineLearningError(f"Unexpected shap_values shape: {shap_values.shape}")
        else:
            raise MachineLearningError("Unexpected type for shap_values")

    @abstractmethod
    def run_method(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Main method to be implemented by subclasses"""
        pass
