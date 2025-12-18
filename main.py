import logging
import yaml
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder
from steps.ingest import Ingestion
from steps.train import Trainer

logging.basicConfig(level=logging.INFO)

def train_with_mlflow():

    with open("config.yml") as f:
        config = yaml.safe_load(f)

    mlflow.set_experiment("Mushroom Classification")

    with mlflow.start_run():

        # Load data
        ingestion = Ingestion()
        train_df, test_df = ingestion.load_data()

        X_train = train_df.drop(columns=["class"])
        y_train = train_df["class"]

        X_test = test_df.drop(columns=["class"])
        y_test = test_df["class"]

        # Encode label
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        # Train
        trainer = Trainer()
        pipeline = trainer.train_and_save(X_train, y_train)

        # Evaluate
        acc = pipeline.score(X_test, y_test)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    train_with_mlflow()
