"""llama.cpp subprocess adapter.

This adapter invokes a local `llama.cpp`-compatible binary via subprocess.
It makes no network calls and requires the user to provide both a local
`model_path` (ggml file) and the path to the `llama.cpp` binary (or rely on PATH).

The adapter is defensive: it probes a few common CLI argument patterns,
uses a timeout, and falls back to `SimpleStubLLM` when invocation fails.

Usage:
    adapter = LlamaCppAdapter(model_path="/path/to/model.ggml", binary_path="/usr/local/bin/main", seed=42)
    text = adapter.generate("Hello world")
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

from .base import BaseLLM, SimpleStubLLM

logger = logging.getLogger(__name__)


class LlamaCppAdapter(BaseLLM):
    def __init__(self, model_path: str, binary_path: Optional[str] = None, *, seed: int = 42, timeout: int = 30, extra_args: Optional[List[str]] = None):
        self.model_path = model_path
        self.binary_path = binary_path or "main"
        self.seed = seed
        self.timeout = timeout
        self.extra_args = extra_args or []

        self._binary = self._find_binary()

    def _find_binary(self) -> Optional[str]:
        # If binary_path is an absolute/relative path and exists, use it.
        if shutil.which(self.binary_path):
            path = shutil.which(self.binary_path)
            logger.info("LlamaCppAdapter: using binary at %s", path)
            return path
        logger.warning("LlamaCppAdapter: binary '%s' not found on PATH", self.binary_path)
        return None

    def _try_invocations(self, prompt: str) -> Optional[str]:
        # Write prompt to temp file for invocations that accept a file
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=True) as tf:
            tf.write(prompt)
            tf.flush()

            candidates = []

            # Common patterns: --model/--prompt, -m/-p, -m -f (file), --model --file
            candidates.append([self._binary, "--model", self.model_path, "--prompt", prompt])
            candidates.append([self._binary, "-m", self.model_path, "-p", prompt])
            candidates.append([self._binary, "--model", self.model_path, "--prompt-file", tf.name])
            candidates.append([self._binary, "-m", self.model_path, "-f", tf.name])
            # Some builds support interactive with stdin
            candidates.append([self._binary, "--model", self.model_path, "--interactive-start"])

            # Append any extra args user supplied
            if self.extra_args:
                candidates = [c + self.extra_args for c in candidates]

            for cmd in candidates:
                try:
                    if "-f" in cmd or "--prompt-file" in cmd:
                        # prompt passed via file; run without stdin
                        logger.debug("LlamaCppAdapter: attempting command: %s", " ".join(cmd))
                        proc = subprocess.run(cmd, capture_output=True, timeout=self.timeout, check=False)
                    else:
                        # send prompt via stdin for interactive-style invocations
                        logger.debug("LlamaCppAdapter: attempting command (stdin): %s", " ".join(cmd))
                        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, timeout=self.timeout, check=False)

                    out = proc.stdout.decode("utf-8", errors="ignore").strip()
                    err = proc.stderr.decode("utf-8", errors="ignore").strip()
                    if proc.returncode == 0 and out:
                        logger.info("LlamaCppAdapter: invocation succeeded with command: %s", cmd[:3])
                        return out
                    else:
                        logger.debug("LlamaCppAdapter: command failed (rc=%s). stderr=%s", proc.returncode, err)
                except FileNotFoundError:
                    logger.debug("LlamaCppAdapter: binary not found for command: %s", cmd[0])
                except subprocess.TimeoutExpired:
                    logger.warning("LlamaCppAdapter: invocation timed out for command: %s", cmd[:3])
                except Exception as e:
                    logger.exception("LlamaCppAdapter: unexpected error invoking command: %s", e)

        return None

    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> str:
        # If no binary available, fallback immediately
        if not self._binary:
            logger.warning("LlamaCppAdapter: no binary available; falling back to stub")
            return SimpleStubLLM(seed=self.seed).generate(prompt)

        # Insert seed/ determinism where possible via env or args (many builds accept --seed)
        try:
            # add seed arg to extra args for invocation attempts
            seed_arg = ["--seed", str(self.seed)]
            if seed_arg not in self.extra_args:
                self.extra_args = self.extra_args + seed_arg

            out = self._try_invocations(prompt)
            if out is None:
                logger.warning("LlamaCppAdapter: all invocations failed; using stub fallback")
                return SimpleStubLLM(seed=self.seed).generate(prompt)
            return out
        except Exception as e:
            logger.exception("LlamaCppAdapter: generation failed: %s", e)
            return SimpleStubLLM(seed=self.seed).generate(prompt)
