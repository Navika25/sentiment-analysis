import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

# Title
st.title("💬 Sentiment Analysis App 😊")
st.write("Analyze whether your text is Positive or Negative")

# ✅ Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    return df

df = load_data()

# ✅ Train Model
@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["review"])
    
    model = MultinomialNB()
    model.fit(X, df["sentiment"])
    
    return vectorizer, model

vectorizer, model = train_model()

# ✅ Input Box
user_input = st.text_input("✍️ Enter your review:")

# ✅ Button
if st.button("🔍 Analyze"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😡 Negative Sentiment")
    else:
        st.warning("⚠️ Please enter text")