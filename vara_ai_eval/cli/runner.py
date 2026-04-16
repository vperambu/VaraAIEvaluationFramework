import argparse
import logging
from ..logging_config import setup_logger
from ..models.base import SimpleStubLLM
from ..models.llama_cpp_adapter import LlamaCppAdapter
from ..models.llama_adapter import LlamaAdapter
from ..retriever.faiss_retriever import FaissRetriever
from ..rag.rag import RAG
from ..evaluator.metrics import Evaluator
from ..guardrails.policy import GuardrailPolicy

logger = setup_logger("vara.cli", logging.INFO)


def main(argv: list | None = None):
    parser = argparse.ArgumentParser(description="Vara AI Evaluation runner")
    parser.add_argument("--config", required=False, help="Path to config (optional)")
    parser.add_argument("--llama-cpp-binary", required=False, help="Path or name of llama.cpp binary (optional)")
    parser.add_argument("--model-path", required=False, help="Path to local model file for adapters (ggml for llama.cpp or transformers local folder)")
    parser.add_argument("--use-transformers", action="store_true", help="Prefer local transformers adapter when a local model path is provided")
    parser.add_argument("--seed", required=False, type=int, default=42, help="Deterministic seed")
    parser.add_argument("--k", required=False, type=int, default=3, help="Number of docs to retrieve")
    parser.add_argument("--json-output", action="store_true", help="Print results as JSON to stdout")

    args = parser.parse_args(argv)

    # Choose model: prefer transformers adapter when requested, otherwise try llama.cpp adapter.
    model = SimpleStubLLM(seed=args.seed)
    if args.use_transformers and args.model_path:
        try:
            model = LlamaAdapter(model_path=args.model_path, device="cpu", seed=args.seed)
            logger.info("Using LlamaAdapter (transformers) with model=%s", args.model_path)
        except Exception as e:
            logger.exception("Failed to initialize LlamaAdapter, falling back: %s", e)
    elif args.llama_cpp_binary and args.model_path:
        try:
            model = LlamaCppAdapter(model_path=args.model_path, binary_path=args.llama_cpp_binary, seed=args.seed)
            logger.info("Using LlamaCppAdapter with binary=%s, model=%s", args.llama_cpp_binary, args.model_path)
        except Exception as e:
            logger.exception("Failed to initialize LlamaCppAdapter, falling back to stub: %s", e)

    # Simple embedder that returns fixed vector for determinism in CI.
    def embed_fn(text: str):
        return [1.0] * 32

    # Prepare document store and retriever
    from ..retriever.document_store import DocumentStore
    docstore = DocumentStore(embed_fn=embed_fn, index_path=None)
    # Example: add a tiny document set; real usage should load documents from files
    docstore.add_documents([
        {"id": "d1", "text": "Privacy policy: We do not store passwords.", "meta": {}},
        {"id": "d2", "text": "Terms: Users agree to share non-sensitive info.", "meta": {}},
    ])
    docstore.build_index()

    retriever = FaissRetriever(embed_fn=embed_fn, docstore=docstore)
    rag = RAG(model, retriever, seed=args.seed)

    policy = GuardrailPolicy(banned_patterns=[r"\bpassword\b", r"\bsecret\b"])
    evaluator = Evaluator()

    query = "What is the privacy policy?"
    # enforce guardrails on incoming query
    safe_query = policy.enforce(query)
    result = rag.answer(safe_query, k=args.k)
    metrics = evaluator.evaluate(result["response"], result["docs"])

    if args.json_output:
        import json
        out = {"query": query, "response": result["response"], "docs": result["docs"], "metrics": metrics}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        logger.info("Query: %s", query)
        logger.info("Response: %s", result["response"])
        logger.info("Docs: %s", result["docs"])
        logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
