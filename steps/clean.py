import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Cleaner:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def clean_data(self, data: pd.DataFrame):
        """
        Clean and preprocess mushroom dataset
        """
        # target column
        target_col = "class"

        X = data.drop(columns=[target_col])
        y = data[target_col]

        # one-hot encode categorical features
        X_encoded = pd.get_dummies(X)

        # encode target
        y_encoded = self.label_encoder.fit_transform(y)

        return X_encoded, y_encoded
