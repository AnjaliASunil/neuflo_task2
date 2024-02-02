# frontend.py
import streamlit as st
from backend import SentimentAnalysisModel

def main():
    st.title("Sentiment Analysis App")

    user_input = st.text_area("Enter text for sentiment analysis:")
    if st.button("Submit"):
        # Backend processing
        backend_model = SentimentAnalysisModel()

        # Example training data (replace this with your actual training data)
        X_train = ["positive text 1", "negative text 2", "neutral text 3"]
        y_train = [1, 0, 2]  # Assuming binary labels (1 for positive, 0 for negative, 2 for neutral)

        backend_model.train_model(X_train, y_train)
        result = backend_model.predict_sentiment(user_input)

        # Display result
        st.write(f"Sentiment: {'Positive' if result == 1 else 'Negative' if result == 0 else 'Neutral'}")

if __name__ == "__main__":
    main()
