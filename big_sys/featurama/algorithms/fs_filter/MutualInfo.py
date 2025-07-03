from ..BaseFSFilter import BaseFeatureSelectorFilter
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class MutualInfo(BaseFeatureSelectorFilter):
    """Select top k features based on mutual information"""

    def run_method(self) -> None:
        print("RUN: MutualInfo")
        X = self._get_features_data()
        y = self._get_target_data()

        k = float(self.params.get('k', 0.5))
        k = int(k * X.shape[1])

        print(f"X_features: {X.columns.to_list()}")
        print(f"X_features (len): {len(X.columns.to_list())}")
        print(f"y_feature: {y.name}")
        print(f"params: {self.params}")

        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()]
        print(f"selected_columns: {selected_columns.tolist()}")
        print(f"selected_columns (len): {len(selected_columns.tolist())}")

        return self._save_result(selected_columns.tolist())
