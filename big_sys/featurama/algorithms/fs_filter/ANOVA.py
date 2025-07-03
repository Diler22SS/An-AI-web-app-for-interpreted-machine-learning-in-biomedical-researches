from ..BaseFSFilter import BaseFeatureSelectorFilter
from sklearn.feature_selection import SelectKBest, f_classif


class ANOVA(BaseFeatureSelectorFilter):
    """Select top k features based on ANOVA F-value"""

    def run_method(self) -> None:
        print("RUN: ANOVA")
        X = self._get_features_data()
        y = self._get_target_data()
        X_scaled = self._scale_features(X)

        k = float(self.params.get('k', 0.5))
        k = int(k * X.shape[1])

        print(f"X_features: {X.columns.to_list()}")
        print(f"X_features (len): {len(X.columns.to_list())}")
        print(f"y_feature: {y.name}")
        print(f"params: {self.params}")

        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_scaled, y)
        selected_columns = X.columns[selector.get_support()]
        print(f"selected_columns: {selected_columns.tolist()}")
        print(f"selected_columns (len): {len(selected_columns.tolist())}")

        return self._save_result(selected_columns.tolist())
