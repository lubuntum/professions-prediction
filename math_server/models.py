from __future__ import annotations

from pydantic import Field

from models import ApiModel


class MathPrediction(ApiModel):
    pupil_id: int = Field(alias="pupilId")
    percent: float
    recommendation: str
    scores: dict[str, float]