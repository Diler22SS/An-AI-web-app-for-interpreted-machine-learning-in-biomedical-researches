from ..BaseFSWrapper import BaseFeatureSelectorWrapper, FeatureSelectionError
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression


class SFSLogreg(BaseFeatureSelectorWrapper):
    """Perform forward feature selection with Logistic Regression."""

    def run_method(self) -> None:
        print("RUN: SFSLogreg")
        X = self._get_features_data()
        y = self._get_target_data() 
        X_scaled = self._scale_features(X)

        model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='liblinear',
            max_iter=1000,
            random_state=42,
        )

        scoring = self.params.get('scoring', 'accuracy')
        print(f"X_features: {X.columns.to_list()}")
        print(f"X_features (len): {len(X.columns.to_list())}")
        print(f"y_feature: {y.name}")
        print(f"params: {self.params}")

        if scoring not in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            raise FeatureSelectionError("Invalid scoring metric")

        sfs = SFS(
                model,
                k_features='best',
                forward=True,
                floating=False,
                scoring=scoring,
                verbose=2,
                cv=5
            )
        sfs.fit(X_scaled.values, y.values)
        selected_features = list(X.columns[list(sfs.k_feature_idx_)])
        print(f"selected_features: {selected_features}")
        print(f"selected_features (len): {len(selected_features)}")

        return self._save_result(selected_features)
