from ..BaseMLModel import BaseMachineLearning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier


class XGBLinear(BaseMachineLearning):
    """Building XGBLinear model."""

    def run_method(self) -> None:
        print("RUN: XGBLinear")
        X = self._get_features_data()
        y = self._get_target_data()
        test_size = float(self.params.get('test_size', 0.2))

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        X_train_scaled = self._scale_features(X_train)
        X_test_scaled = self._scale_features(X_test)

        n_estimators = int(self.params.get('n_estimators', 100))
        learning_rate = float(self.params.get('learning_rate', 0.3))

        print(f"X_features: {X.columns.to_list()}")
        print(f"X_features (len): {len(X.columns.to_list())}")
        print(f"y_feature: {y.name}")
        print(f"params: {self.params}")

        model = XGBClassifier(
            booster='gblinear',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric='auc',
            random_state=42,
        )

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"roc_auc: {roc_auc}")
        print(f"accuracy: {accuracy}")
        print(f"f1: {f1}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")

        explainer, shap_values = self.shap_linear_explainer(model, X_test_scaled)
        global_plot = self._generate_global_shap_plot(X_test_scaled, shap_values)
        distribution_plot = self._generate_distribution_shap_plot(X_test_scaled, shap_values)

        return self._save_result(roc_auc, accuracy, f1, precision, recall,
                                 explainer, shap_values,
                                 global_plot, distribution_plot)
