"""FastAPI service for the spam model. Run: uvicorn app:app --reload"""
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

model = joblib.load("model.joblib")
app = FastAPI(title="SMS Spam API")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(msg: Message):
    label = int(model.predict([msg.text])[0])
    return {"label": "spam" if label else "ham"}
