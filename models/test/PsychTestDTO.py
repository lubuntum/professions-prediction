from pydantic import BaseModel
from typing import Optional, List, Dict

from models.test.PsychParamDTO import PsychParamDTO


class PsychTestDTO(BaseModel):
    completionTimeSeconds: Optional[float]
    psychParams: List[PsychParamDTO]
    testTypeName: str
    createdAt: Optional[str]