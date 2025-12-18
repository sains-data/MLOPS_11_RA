import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class Cleaner:
    def __init__(self):
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        self.columns_ = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoder on training data and transform it
        """
        X_encoded = self.encoder.fit_transform(X)

        self.columns_ = self.encoder.get_feature_names_out(X.columns)

        return pd.DataFrame(
            X_encoded,
            columns=self.columns_,
            index=X.index
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted encoder
        """
        X_encoded = self.encoder.transform(X)

        return pd.DataFrame(
            X_encoded,
            columns=self.columns_,
            index=X.index
        )
