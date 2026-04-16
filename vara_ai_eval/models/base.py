from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLLM(ABC):
    """Model-agnostic LLM adapter interface.

    Implementations must be deterministic when a `seed` is provided.
    """

    @abstractmethod
    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError()


class SimpleStubLLM(BaseLLM):
    """A deterministic, test-friendly stub used for unit tests and CI.

    This is NOT a production model adapter — real adapters should subclass
    `BaseLLM` and implement `generate` to call local models (e.g., Llama).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Deterministic echo with seed-based stable suffix for tests
        suffix = f" [stub-seed={self.seed}]"
        return (prompt or "") + suffix
