from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PsychParam(ApiModel):
    name: str = Field(min_length=1)
    value: float | None = Field(alias="param")


class PsychTest(ApiModel):
    completion_time_seconds: float | None = Field(default=None, alias="completionTimeSeconds")
    psych_params: list[PsychParam] = Field(alias="psychParams")
    test_type_name: str = Field(alias="testTypeName", min_length=1)
    created_at: datetime | None = Field(default=None, alias="createdAt")


class Pupil(ApiModel):
    pupil_id: int = Field(alias="pupilId", gt=0)
    psych_tests: dict[str, PsychTest] = Field(alias="psychTests")
    age: int = Field(alias="age")


class Specialist(ApiModel):
    specialist_id: int = Field(alias="specialistId", gt=0)
    profession: str = Field(min_length=1)
    psych_tests: dict[str, PsychTest] = Field(alias="psychTests")


class Prediction(ApiModel):
    pupil_id: int = Field(alias="pupilId")
    cluster: int
    predicted_profession: str = Field(alias="predictedProfession")
    nearest_specialist_id: int = Field(alias="nearestSpecialistId")
    distance: float
    confidence_category: str = Field(alias="confidenceCategory")


class Health(ApiModel):
    status: str
    clusters: str = Field(alias="referenceData")
    clusters_updated_at: datetime | None = Field(alias="lastSuccessfulRefresh")
