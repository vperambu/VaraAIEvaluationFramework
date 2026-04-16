from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    seed: int = 42
    model_name: Optional[str] = None
    device: str = "cpu"
    faiss_index_path: Optional[str] = None
    verbose: bool = False
