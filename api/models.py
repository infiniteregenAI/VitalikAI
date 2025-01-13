from typing import List, Dict, Any
from pydantic import BaseModel

class ReasoningLayer(BaseModel):
    """Structure for each reasoning layer"""
    name: str
    description: str
    thought_process: str
    conclusion: str
    confidence: float
    sources: List[Dict[str, Any]]