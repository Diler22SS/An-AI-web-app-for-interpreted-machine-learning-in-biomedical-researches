from ..BaseFSFilter import BaseFeatureSelectorFilter
from sklearn.feature_selection import VarianceThreshold


class VarianceThreshold_(BaseFeatureSelectorFilter):
    """Remove features with variance below a specified threshold"""

    def run_method(self) -> None:
        print("RUN: VarianceThreshold_")
        X = self._get_features_data()
        y = self._get_target_data()

        threshold = float(self.params.get('threshold', 0.1))

        print(f"X_features: {X.columns.to_list()}")
        print(f"X_features (len): {len(X.columns.to_list())}")
        print(f"y_feature: {y.name}")
        print(f"params: {self.params}")

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        print(f"selected_columns: {selected_columns.tolist()}")
        print(f"selected_columns (len): {len(selected_columns.tolist())}")

        return self._save_result(selected_columns.tolist())
