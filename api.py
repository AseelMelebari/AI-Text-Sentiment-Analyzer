from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "sentiment_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")


app = FastAPI(
    title="AI Text Sentiment Analyzer API",
    version="1.0.0"
)


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Sentiment Analyzer API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: SentimentRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    transformed_text = vectorizer.transform([text])

    prediction = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]

    prediction_text = (
        "Positive"
        if str(prediction).lower() in ["1", "positive", "pos"]
        else "Negative"
    )

    confidence = float(max(probabilities))

    return {
        "text": text,
        "prediction": prediction_text,
        "confidence": round(confidence, 4),
        "model_name": "TF-IDF Logistic Regression"
    }
