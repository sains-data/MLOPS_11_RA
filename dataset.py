import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class MushroomDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def preprocess(self):
        # target column (edible / poisonous)
        target_col = "class"

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # encode categorical features
        X_encoded = pd.get_dummies(X)

        # encode target
        y_encoded = LabelEncoder().fit_transform(y)

        return X_encoded, y_encoded
