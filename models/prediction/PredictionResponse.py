from pydantic import BaseModel


class PredictionResponse(BaseModel):
    pupilId: int
    cluster: int
    predictedProfession: str
    nearestSpecialistId: int
    distance: float
    confidenceCategory: str