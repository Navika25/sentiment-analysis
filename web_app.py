import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

# 🌸 AESTHETIC CSS
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
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
    background: linear-gradient(to right, #89f7fe, #66a6ff);
    color: black;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

/* Button hover */
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #66a6ff, #89f7fe);
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
                "<div style='background-color:#d4edda;padding:15px;border-radius:10px;text-align:center;'>😊 <b>Positive Sentiment</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background-color:#f8d7da;padding:15px;border-radius:10px;text-align:center;'>😡 <b>Negative Sentiment</b></div>",
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter text")