from pydantic import BaseModel
from typing import Optional, List, Dict

from models.test.PsychTestDTO import PsychTestDTO


class Pupil(BaseModel):
    pupilId: int
    fullName: str
    psychTests: Dict[str, PsychTestDTO]
