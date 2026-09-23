import os
import re
import joblib
from flask import Flask, request, jsonify, render_template

print("Initializing application...")

# Verify model files exist in the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "sentiment_model.pkl")
tfidf_path = os.path.join(current_dir, "tfidf_vectorizer.pkl")


if not os.path.exists(model_path):
    raise FileNotFoundError(f"Missing file: {model_path}")
if not os.path.exists(tfidf_path):
    raise FileNotFoundError(f"Missing file: {tfidf_path}")

print("Loading saved model and vectorizer...")
model = joblib.load(model_path)
tfidf = joblib.load(tfidf_path)
print("Model and vectorizer loaded successfully!")

app = Flask(__name__)

LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

CONTRACTION_MAP = {
    r"\bdidn't\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bisn't\b": "is not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bcan't\b": "cannot",
    r"\bcouldn't\b": "could not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not"
}

HTML_REGEX = re.compile(r"<.*?>")
URL_REGEX = re.compile(r"http\S+|www\S+")
NON_ALPHA_REGEX = re.compile(r"[^a-z\s]")
MULTI_SPACE_REGEX = re.compile(r"\s+")

def preprocess_text(text):
    text = str(text).lower()
    for pattern, repl in CONTRACTION_MAP.items():
        text = re.sub(pattern, repl, text)
    text = HTML_REGEX.sub(" ", text)
    text = URL_REGEX.sub(" ", text)
    text = NON_ALPHA_REGEX.sub(" ", text)
    return MULTI_SPACE_REGEX.sub(" ", text).strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        raw_review = payload.get("review", "").strip()

        if not raw_review:
            return jsonify({"error": "Please enter a valid review text."}), 400

        cleaned_review = preprocess_text(raw_review)
        transformed_vector = tfidf.transform([cleaned_review])

        pred_code = model.predict(transformed_vector)[0]
        prob = model.predict_proba(transformed_vector)[0][pred_code] * 100

        return jsonify({
            "sentiment": LABEL_MAPPING[pred_code],
            "confidence": f"{prob:.1f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n--------------------------------------------------")
    print("Starting Flask server on http://127.0.0.1:5000 ...")
    print("Press CTRL+C to stop.")
    print("--------------------------------------------------\n")
    # debug=False prevents the reloader child-process quirk on Windows
    app.run(host="127.0.0.1", port=5000, debug=False)