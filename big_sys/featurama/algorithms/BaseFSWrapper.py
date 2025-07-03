# Feature selection algorithms

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from django.db import models


class FeatureSelectionError(Exception):
    """Custom exception for feature selection errors."""
    pass


class BaseFeatureSelectorWrapper(ABC):
    def __init__(self, pipeline: models.Model):
        self.pipeline = pipeline
        self._validate_input()
        self.params = self.pipeline.fs_wrapper_parameters

    def _validate_input(self) -> None:
        if not self.pipeline.data_content:
            raise FeatureSelectionError("Pipeline has no dataset attached")

        if not self.pipeline.target_variable:
            raise FeatureSelectionError("Pipeline has no target variable")

        if not self.pipeline.fs_filter_selected_features:
            raise FeatureSelectionError("Empty filter selected feature list")

        if not self.pipeline.fs_wrapper:
            raise FeatureSelectionError("Wasn't choosen wrapper method")

        if not self.pipeline.fs_wrapper_parameters:
            raise FeatureSelectionError("Empty wrapper method's params")

    def _get_features_data(self) -> pd.DataFrame:
        """Get DataFrame with only filter selected features"""
        df = self.pipeline.get_dataframe()
        return df[self.pipeline.fs_filter_selected_features]

    def _get_target_data(self) -> pd.Series:
        """Get target Series"""
        df = self.pipeline.get_dataframe()
        return df[self.pipeline.target_variable]

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X_scaled

    def _save_result(self, selected_features: list) -> None:
        """Save results"""
        self.pipeline.fs_wrapper_selected_features = selected_features
        self.pipeline.save()

    @abstractmethod
    def run_method(self) -> None:
        """Main feature selection method to be implemented by subclasses"""
        pass
