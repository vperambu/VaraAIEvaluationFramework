"""LlamaAdapter: optional local Llama model adapter.

This adapter is intentionally conservative:
- It prefers a local `transformers` backend and requires `local_files_only=True` to
  avoid automatic downloads.
- It does not assume any network or cloud dependencies.
- If no supported backend is available it fails with an informative error and
  falls back to the deterministic `SimpleStubLLM` at runtime.

To use:
    adapter = LlamaAdapter(model_path="/path/to/local/model", device="cpu", seed=42)
    text = adapter.generate("Hello")

Note: for production you may implement additional backends (llama.cpp subprocess,
ggml, or custom runtimes) by adding them to `_init_backend`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import BaseLLM, SimpleStubLLM

logger = logging.getLogger(__name__)


class LlamaAdapter(BaseLLM):
    """Adapter for local Llama-family models.

    - `model_path` must point to local model files (no automatic downloads).
    - Deterministic generation is attempted by seeding the RNG.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        seed: int = 42,
        backend: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.backend = backend
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 128,
            "do_sample": False,
        }

        self._model = None
        self._tokenizer = None
        self._backend = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Try to initialize a supported local backend.

        Current implementation: `transformers` with `local_files_only=True`.
        """
        # Prefer explicit backend when provided
        if self.backend == "transformers":
            tried_transformers = True
        else:
            tried_transformers = False

        # Try transformers if available or requested
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            tried_transformers = True
        except Exception:
            if self.backend == "transformers":
                raise
            tried_transformers = False

        if tried_transformers:
            try:
                import torch  # type: ignore
                from transformers import (  # type: ignore
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )

                # Require local files only to avoid network calls during CI
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, local_files_only=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, local_files_only=True
                )
                # Move model to device if possible
                try:
                    self._model.to(torch.device(self.device))
                except Exception:
                    logger.debug(
                        "Could not move model to device %s; continuing on default device",
                        self.device,
                    )
                self._backend = "transformers"
                logger.info(
                    "LlamaAdapter: initialized transformers backend (local files only)"
                )
                return
            except Exception as e:
                logger.warning("Transformers backend initialization failed: %s", e)

        # No supported backend found
        raise RuntimeError(
            "No supported local backend found for LlamaAdapter. Install 'transformers' and provide a local model_path, "
            "or implement an alternate backend (e.g., llama.cpp) and set `backend` accordingly."
        )

    def generate(
        self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        if self._model is None or self._tokenizer is None:
            logger.warning("LlamaAdapter not initialized; using SimpleStubLLM fallback")
            return SimpleStubLLM(seed=self.seed).generate(prompt)

        try:
            import torch  # type: ignore

            # Deterministic setup
            try:
                torch.manual_seed(self.seed)
            except Exception:
                logger.debug("torch.manual_seed unavailable or failed; continuing")

            gen = torch.Generator()
            try:
                gen.manual_seed(self.seed)
            except Exception:
                logger.debug("torch.Generator.manual_seed failed; continuing")

            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            gen_kwargs = dict(self.generation_kwargs)
            # Provide generator when supported for deterministic sampling
            gen_kwargs.setdefault("generator", gen)

            out = self._model.generate(**inputs, **gen_kwargs)
            text = self._tokenizer.decode(out[0], skip_special_tokens=True)
            return text
        except Exception as e:
            logger.exception("LlamaAdapter generation failed: %s", e)
            # Safe fallback to stub
            return SimpleStubLLM(seed=self.seed).generate(prompt)
