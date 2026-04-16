import re
from typing import List
import logging

logger = logging.getLogger(__name__)


class GuardrailPolicy:
    """Simple, extendable guardrail checks for prompt injections and unsafe content.

    Rules are lightweight and deterministic; extend with custom regexes or plugins.
    """

    def __init__(self, banned_patterns: List[str] = None):
        self.banned_patterns = [re.compile(p, re.IGNORECASE) for p in (banned_patterns or [])]

    def check(self, text: str) -> List[str]:
        """Return list of matched rule descriptions (empty = pass)."""
        matches = []
        for rx in self.banned_patterns:
            if rx.search(text):
                matches.append(f"pattern:{rx.pattern}")
        if matches:
            logger.debug("Guardrail matched rules: %s", matches)
        return matches

    def enforce(self, text: str) -> str:
        """Apply minimal transformations or redact when needed.

        Default behavior: if banned pattern present, redact the offending spans.
        """
        if not self.banned_patterns:
            return text
        out = text
        for rx in self.banned_patterns:
            out = rx.sub("[REDACTED]", out)
        return out
