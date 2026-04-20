import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment App", page_icon="😊", layout="centered")

# Custom CSS (for styling)
st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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

# Input box
user_input = st.text_input("✍️ Enter your review:")

# Button
if st.button("🔍 Analyze Sentiment"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)

        # Styled output
        if prediction[0] == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")
    else:
        st.warning("⚠️ Please enter some text")