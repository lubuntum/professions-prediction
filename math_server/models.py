from __future__ import annotations

from pydantic import Field

from models import ApiModel


class MathPredictionItem(ApiModel):
    profession: str = Field(alias="profession")
    percentage: float = Field(alias="percentage")
    recommendation: str = Field(alias="recommendation")
    recommendation_complex: str = Field(alias="recommendationComplex")
    aizen_norm: float = Field(alias="aizenNorm")
    belbin_norm: float = Field(alias="belbinNorm")
    bennet_norm: float = Field(alias="bennetNorm")
    final_score: float = Field(alias="finalScore")
    utility: float = Field(alias="utility")


class MathPrediction(ApiModel):
    pupil_id: int = Field(alias="pupilId")
    professions: list[MathPredictionItem] = Field(alias="professions")