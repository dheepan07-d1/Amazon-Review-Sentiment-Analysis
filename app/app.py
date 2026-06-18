from flask import Flask, render_template, request
import pickle
import os

# Flask App
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ----------------------------
# Load Model and TF-IDF
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "sentiment_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)


# ----------------------------
# Home Page
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# Prediction
# ----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)[0]

    prob = model.predict_proba(review_vector)
    confidence = round(max(prob[0]) * 100, 2)

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


# ----------------------------
# Run App
# ----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)