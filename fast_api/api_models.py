from pydantic import BaseModel

class LLMInput(BaseModel):
    context_chunks: list[str]
    question: str
    temperature: float
    top_p: float

class NERInput(BaseModel):
    text: str