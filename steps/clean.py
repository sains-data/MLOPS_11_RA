import numpy as np
from sklearn.impute import SimpleImputer

class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="most_frequent")

    def clean_data(self, data):
        # Replace '?' with NaN
        data = data.replace("?", np.nan)

        # Impute missing values
        data.iloc[:, :] = self.imputer.fit_transform(data)

        return data
