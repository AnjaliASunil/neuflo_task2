# backend.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

class SentimentAnalysisModel:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )
        self.vectorizer_fitted = False

    def fit_vectorizer(self, X_train):
        self.model.named_steps['tfidfvectorizer'].fit(X_train)
        self.vectorizer_fitted = True

    def train_model(self, X_train, y_train):
        if not self.vectorizer_fitted:
            self.fit_vectorizer(X_train)

        self.model.fit(X_train, y_train)

    def predict_sentiment(self, text):
        if not self.vectorizer_fitted:
            raise ValueError("TF-IDF vectorizer is not fitted. Please call 'train_model' first.")

        # Additional debugging information
        print("Processed Text:", self.model.named_steps['tfidfvectorizer'].transform([text]))
        print("Model Coefficients:", self.model.named_steps['logisticregression'].coef_)

        return self.model.predict([text])[0]
