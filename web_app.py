import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

# 🌈 AESTHETIC + ANIMATED BACKGROUND
st.markdown("""
<style>

/* 🌈 Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #e0c3fc, #8ec5fc, #f9f9f9, #d4fc79);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

/* Animation */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* 🧊 Glass effect card */
.main {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Title */
h1 {
    text-align: center;
    color: #333;
}

/* Subtitle */
p {
    text-align: center;
    font-size: 18px;
    color: #555;
}

/* Input box */
.stTextInput>div>div>input {
    font-size: 18px;
    padding: 12px;
    border-radius: 12px;
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #667eea, #764ba2);
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
}

/* Button hover */
.stButton>button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("💬 Sentiment Analysis App 😊")
st.write("Analyze whether your text is Positive or Negative")

# ✅ Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/imdb.csv")

df = load_data()

# ✅ Train model
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
                "<div style='background-color:#d4edda;padding:15px;border-radius:12px;text-align:center;font-size:18px;'>😊 <b>Positive Sentiment</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#f8d7da;padding:15px;border-radius:12px;text-align:center;font-size:18px;'>😡 <b>Negative Sentiment</b></div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter text")