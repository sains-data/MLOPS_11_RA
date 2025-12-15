import logging
import yaml
import mlflow
import mlflow.sklearn
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def train_with_mlflow():

    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment("Mushroom Classification Experiment")

    with mlflow.start_run():
        # Ingest
        ingestion = Ingestion()
        train_df, test_df = ingestion.load_data()
        logging.info("Data ingestion completed")

        # Clean (returns X, y)
        cleaner = Cleaner()
        X_train, y_train = cleaner.clean_data(train_df)
        X_test, y_test = cleaner.clean_data(test_df)
        logging.info("Data cleaning completed")

        # Train
        trainer = Trainer()
        trainer.train_model(X_train, y_train)
        trainer.save_model()
        logging.info("Model training completed")

        # Evaluate
        predictor = Predictor()
        accuracy, report, roc_auc = predictor.evaluate_model(X_test, y_test)

        # MLflow logging
        mlflow.set_tag("dataset", "mushroom")
        mlflow.log_params(config['model']['params'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(trainer.pipeline, "model")

        print("\n===== Model Evaluation =====")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"ROC AUC  : {roc_auc:.4f}")
        print(report)
        print("============================\n")

if __name__ == "__main__":
    train_with_mlflow()
