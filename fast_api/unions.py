from pydantic import BaseModel
from typing import List, Union

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]