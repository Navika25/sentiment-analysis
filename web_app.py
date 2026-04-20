from flask import Flask, render_template, request
from src.preprocessing import load_data, clean_data
from src.feature_engineering import vectorize_text
from src.ml_model import train_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Train model once
data = load_data()
data = clean_data(data)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    data['review'], data['sentiment'], test_size=0.2
)

vectorizer, X_train_vec = vectorize_text(X_train)
model = train_model(X_train_vec, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    
    if request.method == "POST":
        text = request.form["text"]
        text_vec = vectorizer.transform([text])
        pred = model.predict(text_vec)

        if pred[0] == 1:
            prediction = "Positive 😊"
        else:
            prediction = "Negative 😡"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")