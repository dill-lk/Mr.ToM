"""
tests/test_rag.py — Unit tests for rmoe.rag (BM25 Vector RAG Engine).

Paper §3.1: "ARLL cross-references all findings against gold-standard
benchmarks (MIMIC-CXR, RSNA Bone Age, clinical guidelines)."
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rmoe.rag import VectorRAGEngine


@pytest.fixture
def rag():
    return VectorRAGEngine()


def test_retrieval_returns_results(rag):
    """Any medical query should return at least one result."""
    refs = rag.get_references("lung nodule CXR", top_k=3)
    assert len(refs) >= 1


def test_retrieval_top_k(rag):
    """get_references respects top_k parameter."""
    refs = rag.get_references("fracture bone X-ray", top_k=2)
    assert len(refs) <= 2


def test_retrieval_lung_nodule_relevant(rag):
    """Query 'pulmonary nodule' should surface a chest / nodule entry."""
    refs = rag.get_references("pulmonary nodule spiculated lung mass", top_k=5)
    combined = " ".join(refs).lower()
    assert any(kw in combined for kw in ("lung", "nodule", "chest", "cxr", "pulmonary", "fleischner"))


def test_retrieval_fracture_relevant(rag):
    """Query 'rib fracture' should surface a fracture / MSK entry."""
    refs = rag.get_references("rib fracture cortical disruption", top_k=5)
    combined = " ".join(refs).lower()
    assert any(kw in combined for kw in ("fracture", "bone", "cortical", "msk", "rib"))


def test_retrieval_empty_query_no_crash(rag):
    """Empty query must not raise an exception."""
    refs = rag.get_references("", top_k=3)
    assert isinstance(refs, list)


def test_retrieval_top1_is_string(rag):
    """Every returned reference should be a non-empty string."""
    refs = rag.get_references("adenocarcinoma lung cancer staging", top_k=3)
    for r in refs:
        assert isinstance(r, str)
        assert len(r) > 0


def test_retrieval_score_ordering(rag):
    """Repeated identical queries return identical results (deterministic)."""
    query = "tuberculosis upper lobe reactivation"
    r1 = rag.get_references(query, top_k=3)
    r2 = rag.get_references(query, top_k=3)
    assert r1 == r2
