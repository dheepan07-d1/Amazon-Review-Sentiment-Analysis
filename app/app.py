
from flask import Flask, render_template, request
import pickle

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# Load Model
model = pickle.load(open("../models/sentiment_model.pkl", "rb"))
tfidf = pickle.load(open("../models/tfidf_vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    # Convert to TF-IDF
    review_vector = tfidf.transform([review])

    # Prediction
    prediction = model.predict(review_vector)[0]

    # Confidence Score
    prob = model.predict_proba(review_vector)
    confidence = round(max(prob[0]) * 100, 2)

    # Emoji
    if prediction == "Positive":
        emoji = "😊"
    elif prediction == "Negative":
        emoji = "😞"
    else:
        emoji = "😐"

    return render_template(
        "index.html",
        review=review,
        prediction=prediction,
        confidence=confidence,
        emoji=emoji
    )


if __name__ == "__main__":
    app.run(debug=True)

