import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self):
        self.config = self.load_config()
        self.model_path = self.config["model"]["store_path"]
        self.model_bundle = self.load_model()
        self.pipeline = self.model_bundle["model"]
        self.columns = self.model_bundle["columns"]

    def load_config(self):
        with open("config.yml", "r") as file:
            return yaml.safe_load(file)

    def load_model(self):
        model_file = os.path.join(self.model_path, "model.pkl")
        return joblib.load(model_file)

    def evaluate_model(self, X_test, y_test):
        # Align columns
        X_test = X_test.reindex(columns=self.columns, fill_value=0)

        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        return accuracy, report, roc_auc
