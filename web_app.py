import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

# 🌸 PASTEL ANIMATED BACKGROUND
st.markdown("""
<style>

/* 🌈 Pastel Gradient Animation */
.stApp {
    background: linear-gradient(-45deg, 
        #ffd6e0,  /* pink */
        #fff5ba,  /* yellow */
        #cce7ff,  /* blue */
        #e6ccff,  /* purple */
        #d4f8e8   /* green */
    );
    background-size: 400% 400%;
    animation: pastelBG 12s ease infinite;
}

/* Animation */
@keyframes pastelBG {
    0% {background-position: 0% 50%;}
    25% {background-position: 50% 100%;}
    50% {background-position: 100% 50%;}
    75% {background-position: 50% 0%;}
    100% {background-position: 0% 50%;}
}

/* 🧊 Glass Card */
.main {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
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
    border: 1px solid #ddd;
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #ff9a9e, #fad0c4);
    color: black;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

/* Hover */
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #a1c4fd, #c2e9fb);
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("💬 Sentiment Analysis App 😊")
st.write("Analyze whether your text is Positive or Negative")

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
                "<div style='background-color:#e0f7e9;padding:15px;border-radius:12px;text-align:center;font-size:18px;'>😊 <b>Positive Sentiment</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#ffe0e0;padding:15px;border-radius:12px;text-align:center;font-size:18px;'>😡 <b>Negative Sentiment</b></div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter text")