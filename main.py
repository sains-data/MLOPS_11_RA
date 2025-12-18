import logging
import yaml
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import LabelEncoder

from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def train_with_mlflow():

    # Load config
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Mushroom Classification Experiment")

    with mlflow.start_run():

        # ======================
        # 1. Data Ingestion
        # ======================
        ingestion = Ingestion()
        train_df, test_df = ingestion.load_data()
        logging.info("Data ingestion completed")

        # ======================
        # 2. Feature & Target Split
        # ======================
        X_train = train_df.drop(columns=["class"])
        y_train = train_df["class"]

        X_test = test_df.drop(columns=["class"])
        y_test = test_df["class"]

        # ======================
        # 3. Encode Target
        # ======================
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # ======================
        # 4. Clean Features
        # ======================
        cleaner = Cleaner()
        X_train = cleaner.clean_data(X_train)
        X_test = cleaner.clean_data(X_test)
        logging.info("Data cleaning completed")

        # ======================
        # 5. Train Model
        # ======================
        trainer = Trainer()
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed")

        # ======================
        # 6. Evaluate Model
        # ======================
        predictor = Predictor()
        accuracy, report, roc_auc = predictor.evaluate_model(X_test, y_test)

        # ======================
        # 7. MLflow Logging
        # ======================
        mlflow.set_tag("dataset", "mushroom")
        mlflow.set_tag("model", config["model"]["name"])

        mlflow.log_params(config["model"]["params"])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        # ======================
        # 8. Output
        # ======================
        print("\n===== Model Evaluation =====")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        print(report)
        print("============================\n")


if __name__ == "__main__":
    train_with_mlflow()
