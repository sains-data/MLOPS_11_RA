import os
import joblib
import yaml
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

class Predictor:
    def __init__(self):
        self.config = self.load_config()
        self.model_path = self.config['model']['store_path']
        self.model, self.columns = self.load_model()

    def load_config(self):
        with open('config.yml', 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_model(self):
        model_file_path = os.path.join(self.model_path, 'model.pkl')
        artifact = joblib.load(model_file_path)
        return artifact["model"], artifact["columns"]

    def align_features(self, X):
        return X[self.columns]
    
    def feature_target_separator(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def evaluate_model(self, X_test, y_test):
        X_test = self.align_features(X_test)
    
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
    
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
    
        return accuracy, class_report, roc_auc

