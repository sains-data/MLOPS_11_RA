import numpy as np
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent")

    def clean_data(self, X):
        X = X.replace("?", np.nan)
        X.iloc[:, :] = self.imputer.fit_transform(X)
        return X
