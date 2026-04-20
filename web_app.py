import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("Sentiment Analysis App 😊")

data = {
    "text": [
        "I love this movie",
        "This is amazing",
        "I hate this",
        "This is bad"
    ],
    "sentiment": ["positive", "positive", "negative", "negative"]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["sentiment"])

user_input = st.text_input("Enter your review:")

if st.button("Analyze"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.warning("Please enter text")