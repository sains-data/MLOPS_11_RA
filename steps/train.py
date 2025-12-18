import os
import joblib
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class Trainer:
    def __init__(self, config_path="config.yml"):
        self.config = self.load_config(config_path)
        self.model_path = self.config["model"]["store_path"]
        self.model_params = self.config["model"]["params"]

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def create_pipeline(self, X):
        """
        Create preprocessing + model pipeline
        All features are categorical (mushroom dataset)
        """
        categorical_cols = X.columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_cols
                )
            ]
        )

        model = RandomForestClassifier(**self.model_params)

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model)
            ]
        )

        return pipeline

    def train_and_save(self, X_train, y_train):
        pipeline = self.create_pipeline(X_train)
        pipeline.fit(X_train, y_train)

        os.makedirs(self.model_path, exist_ok=True)

        joblib.dump(
            {
                "model": pipeline,
                "columns": X_train.columns.tolist()
            },
            os.path.join(self.model_path, "model.pkl")
        )

        return pipeline
