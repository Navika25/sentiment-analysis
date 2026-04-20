import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Sentiment App", page_icon="😊")

# Title
st.title("💬 Sentiment Analysis App 😊")
st.write("Analyze whether your text is Positive or Negative")

# Sample data
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

# Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

model = MultinomialNB()
model.fit(X, df["sentiment"])

# ✅ INPUT BOX (THIS WAS MISSING)
user_input = st.text_input("✍️ Enter your review here:")

# Button
if st.button("🔍 Analyze"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)

        if prediction[0] == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")
    else:
        st.warning("⚠️ Please enter text")