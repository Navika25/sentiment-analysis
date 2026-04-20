import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="centered")

# 🎨 CUSTOM CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    .main {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.2);
        text-align: center;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(to right, #ff7e5f, #feb47b);
        color: white;
        font-size: 18px;
        padding: 10px 25px;
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>💬 Sentiment Analysis App 😊</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyze whether your text is Positive or Negative</p>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/imdb.csv")

df = load_data()

# Train model
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["review"])
    model = MultinomialNB()
    model.fit(X, df["sentiment"])
    return vectorizer, model

vectorizer, model = train_model()

# Input
user_input = st.text_input("✍️ Enter your review:")

# Button
if st.button("🔍 Analyze"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]

        if prediction == "positive":
            st.markdown(
                "<div style='background-color:#d4edda;padding:15px;border-radius:10px;'>😊 <b>Positive Sentiment</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#f8d7da;padding:15px;border-radius:10px;'>😡 <b>Negative Sentiment</b></div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter text")