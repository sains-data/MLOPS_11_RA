import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Cleaner:
    def __init__(self):
        self.feature_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        self.target_encoder = LabelEncoder()
        self.fitted = False

    def clean_data(self, data: pd.DataFrame):
        data = data.replace("?", np.nan)

        y = data["class"]
        X = data.drop(columns=["class"])

        y_encoded = self.target_encoder.fit_transform(y)

        if not self.fitted:
            X_encoded = self.feature_encoder.fit_transform(X)
            self.fitted = True
        else:
            X_encoded = self.feature_encoder.transform(X)

        feature_names = self.feature_encoder.get_feature_names_out(X.columns)
        X_encoded = pd.DataFrame(X_encoded, columns=feature_names)

        return X_encoded, y_encoded
