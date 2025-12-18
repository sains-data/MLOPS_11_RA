import os
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class Trainer:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model_path = self.config['model']['store_path']
        self.model_params = self.config['model']['params']
        self.pipeline = self.create_pipeline()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def create_pipeline(self):
        """
        Pipeline for mushroom classification
        (features already encoded in clean step)
        """
        model = RandomForestClassifier(**self.model_params)

        pipeline = Pipeline([
            ('model', model)
        ])

        return pipeline

    def train_model(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        model_file = os.path.join(self.model_path, "model.pkl")

        joblib.dump({
            "model": self.pipeline,
            "columns": self.columns
        }, model_file)

