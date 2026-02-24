from pydantic import BaseModel


class ReviewRequest(BaseModel):
    text: str


class ReviewResponse(BaseModel):
    text: str
    sentiment: str
