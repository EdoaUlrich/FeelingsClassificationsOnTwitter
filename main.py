from fastapi import FastAPI
from pydantic import BaseModel, Field


class TweetRequest(BaseModel):
    text: str = Field(..., max_length=280)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: str
    probability_positive: float
    probability_negative: float

