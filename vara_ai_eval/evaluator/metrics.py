from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Collection of lightweight, extendable evaluation metrics.

    Implementations should be deterministic and unit-testable.
    """

    def hallucination_score(self, response: str, docs: List[str]) -> float:
        """Return 0.0..1.0 where higher means more hallucinated.

        Heuristic: if response contains tokens not present in docs, score rises.
        This is intentionally simple and a starting point for more advanced checks.
        """
        if not response:
            return 1.0
        # Accept docs as list of strings or list of dicts with 'text'
        if docs and isinstance(docs[0], dict):
            doc_text = "\n".join(d.get("text", "") for d in docs)
        else:
            doc_text = "\n".join(docs or [])
        # naive heuristic: fraction of words in response not in docs
        rsp_words = set(response.lower().split())
        doc_words = set(doc_text.lower().split())
        if not doc_words:
            return 0.5
        unknown = [w for w in rsp_words if w not in doc_words]
        score = min(1.0, len(unknown) / max(1, len(rsp_words)))
        logger.debug("Hallucination heuristic: %d unknown / %d total -> %f", len(unknown), len(rsp_words), score)
        return score

    def grounding_score(self, response: str, docs: List[str]) -> float:
        """Return 0.0..1.0 where higher means better-grounded.

        Heuristic: overlap fraction between response and docs.
        """
        if not response:
            return 0.0
        if docs and isinstance(docs[0], dict):
            doc_text = "\n".join(d.get("text", "") for d in docs)
        else:
            doc_text = "\n".join(docs or [])
        rsp_words = set(response.lower().split())
        doc_words = set(doc_text.lower().split())
        if not rsp_words:
            return 0.0
        overlap = rsp_words.intersection(doc_words)
        score = len(overlap) / len(rsp_words)
        logger.debug("Grounding heuristic: %d overlap / %d total -> %f", len(overlap), len(rsp_words), score)
        return score

    def citation_alignment_score(self, response: str, docs: List[Dict]) -> float:
        """Heuristic score (0..1) measuring whether the response explicitly aligns to provided docs.

        Checks for doc id mentions and presence of longer matching n-grams (3-5 words) from docs.
        """
        if not docs:
            return 0.0
        matched = 0
        total = len(docs)
        rsp = response.lower()
        for d in docs:
            text = (d.get("text") if isinstance(d, dict) else d) or ""
            tid = str(d.get("id")) if isinstance(d, dict) else None
            found = False
            if tid and tid.lower() in rsp:
                found = True
            # check n-grams
            words = text.lower().split()
            for n in range(5, 2, -1):
                for i in range(max(0, len(words) - n + 1)):
                    gram = " ".join(words[i : i + n])
                    if gram and gram in rsp:
                        found = True
                        break
                if found:
                    break
            if found:
                matched += 1
        score = matched / total
        logger.debug("Citation alignment: %d/%d -> %f", matched, total, score)
        return score

    def exactness_score(self, response: str, docs: List[Dict]) -> float:
        """Compute fraction of response sentences that appear verbatim in docs.

        Simple, deterministic heuristic suitable for CI and unit tests.
        """
        if not response:
            return 0.0
        # split into sentences naively
        import re

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response) if s.strip()]
        if not sents:
            return 0.0
        doc_text = "\n".join(d.get("text", "") if isinstance(d, dict) else d for d in (docs or []))
        matched = 0
        for s in sents:
            if s and s in doc_text:
                matched += 1
        score = matched / len(sents)
        logger.debug("Exactness: %d/%d -> %f", matched, len(sents), score)
        return score

    def evaluate(self, response: str, docs: List[str]) -> Dict[str, float]:
        h = self.hallucination_score(response, docs)
        g = self.grounding_score(response, docs)
        # accept docs as dicts or list of strings
        citation = self.citation_alignment_score(response, docs if docs else [])
        exact = self.exactness_score(response, docs if docs else [])
        return {"hallucination": h, "grounding": g, "citation_alignment": citation, "exactness": exact}
