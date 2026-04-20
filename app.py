from src.preprocessing import load_data, clean_data
from src.feature_engineering import vectorize_text
from src.ml_model import train_model
from sklearn.model_selection import train_test_split

# Load and clean data
data = load_data()
data = clean_data(data)

# Convert labels
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2
)

# Feature engineering
vectorizer, X_train_vec = vectorize_text(X_train)

# Train model
model = train_model(X_train_vec, y_train)

print("Model trained successfully 🚀")

# Prediction loop
while True:
    text = input("Enter a sentence: ")
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)

    if prediction[0] == 1:
        print("Prediction: Positive 😊")
    else:
        print("Prediction: Negative 😡")