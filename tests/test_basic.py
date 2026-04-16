from vara_ai_eval.models.base import SimpleStubLLM
from vara_ai_eval.rag.rag import RAG


def test_stub_and_rag():
    model = SimpleStubLLM(seed=123)
    def embed_fn(x):
        return [0.0]*16
    class DummyRetriever:
        def retrieve(self, q, k=3):
            return ["doc1 text", "doc2 text"]
    retriever = DummyRetriever()
    rag = RAG(model, retriever, seed=123)
    out = rag.answer("Hello test", k=2)
    assert "stub-seed=123" in out["response"]
    assert isinstance(out["docs"], list)
