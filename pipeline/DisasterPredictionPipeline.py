# pipeline.py
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

class DisasterPredictionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, n_estimators=700):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.classifier = RandomForestClassifier()
        self.reg_model = xgb.XGBRegressor(objective='count:poisson', n_estimators=self.n_estimators)
        self.non_zero_probs = None

    def fit(self, X_train_binary, y_train_binary, X_train_xgboost, y_train_xgboost):
        # Train the classifier
        self.classifier.fit(X_train_binary, y_train_binary)
        
        # Predict probabilities of being non-zero
        self.non_zero_probs = self.classifier.predict_proba(X_train_xgboost)[:, 1]
        
        # Train the regression model on non-zero data only
        self.reg_model.fit(X_train_xgboost, y_train_xgboost)
        return self

    def predict(self, X_test):
        # Predict probabilities of being non-zero
        non_zero_probs = self.classifier.predict_proba(X_test)[:, 1]
        
        final_predictions = []
        for i, prob in enumerate(non_zero_probs):
            if prob > self.threshold:
                # Predict the disaster count using regression for non-zero cases
                final_predictions.append(self.reg_model.predict(X_test[i:i+1])[0])
            else:
                # Predict zero for cases with low non-zero probability
                final_predictions.append(0)
        return np.round(final_predictions).astype(int)