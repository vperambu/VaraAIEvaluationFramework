from typing import List, Optional
import logging
from ..models.base import BaseLLM

logger = logging.getLogger(__name__)


class RAG:
    """Retriever-Augmented Generation orchestrator.

    Keeps logic model-agnostic and deterministic when seed is provided.
    """

    def __init__(self, model: BaseLLM, retriever, *, seed: int = 42):
        self.model = model
        self.retriever = retriever
        self.seed = seed

    def build_prompt(self, query: str, docs: Optional[List[str]] = None) -> str:
        # if docs are dicts from DocumentStore, extract text field
        if docs and len(docs) and isinstance(docs[0], dict):
            docs_texts = [d.get("text", "") for d in docs]
        else:
            docs_texts = docs or []
        docs_part = "\n\n".join(docs_texts)
        prompt = f"User query:\n{query}\n\nContext:\n{docs_part}\n\nAnswer concisely."
        logger.debug("Built prompt with %d docs", len(docs or []))
        return prompt

    def answer(self, query: str, k: int = 3) -> dict:
        docs = []
        try:
            docs = self.retriever.retrieve(query, k=k)
        except Exception as e:
            logger.exception("Retriever failed: %s", e)

        prompt = self.build_prompt(query, docs)
        try:
            response = self.model.generate(prompt, metadata={"seed": self.seed})
        except Exception as e:
            logger.exception("Model generate failed: %s", e)
            response = ""

        return {"query": query, "response": response, "docs": docs}
