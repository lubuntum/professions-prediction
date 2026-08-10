from pydantic import BaseModel


class PsychParamDTO(BaseModel):
    name: str
    param: float