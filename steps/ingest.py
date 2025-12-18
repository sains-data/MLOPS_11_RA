import pandas as pd
import yaml
from pathlib import Path

class Ingestion:
    def __init__(self, config_path="config.yml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        train_data_path = self.config['data']['train_path']
        test_data_path = self.config['data']['test_path']

        train_data = pd.read_csv(Path(train_data_path))
        test_data = pd.read_csv(Path(test_data_path))

        return train_data, test_data
