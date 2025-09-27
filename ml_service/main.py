from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI(title="Review Radar - ML Service")

# Load sentiment model at startup
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

class Review(BaseModel):
    text: str

class BatchRequest(BaseModel):
    reviews: List[Review]
    top_k: Optional[int] = 10

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze_batch")
async def analyze_batch(body: BatchRequest):
    texts = [r.text.strip() for r in body.reviews if r.text.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No valid review texts provided")

    # 1. Sentiment analysis
    preds = sentiment_pipe(texts, truncation=True)

    results = []
    for t, p in zip(texts, preds):
        results.append({
            "text": t,
            "label": p["label"],  # POSITIVE or NEGATIVE
            "score": float(p["score"])
        })

    # 2. Summarize
    pos = sum(1 for r in results if r["label"].startswith("POS"))
    neg = sum(1 for r in results if r["label"].startswith("NEG"))
    neu = len(results) - pos - neg

    summary = {
        "total": len(results),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positive_pct": round(pos / len(results) * 100, 2),
        "negative_pct": round(neg / len(results) * 100, 2),
        "neutral_pct": round(neu / len(results) * 100, 2),
    }

    # 3. Keywords
    def top_keywords(docs, k):
        if not docs:
            return []
        vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
        X = vec.fit_transform(docs)
        scores = np.asarray(X.sum(axis=0)).ravel()
        idx = scores.argsort()[-k:][::-1]
        feats = vec.get_feature_names_out()
        return [feats[i] for i in idx]

    positive_docs = [r["text"] for r in results if r["label"].startswith("POS")]
    negative_docs = [r["text"] for r in results if r["label"].startswith("NEG")]

    return {
        "summary": summary,
        "per_review": results,
        "top_positive_keywords": top_keywords(positive_docs, body.top_k or 10),
        "top_negative_keywords": top_keywords(negative_docs, body.top_k or 10)
    }
