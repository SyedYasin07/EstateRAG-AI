"""
Tests for budget extraction, hard constraints, and category search.
"""

import pytest
from app.retrieval.query_parser import QueryParser
from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.hybrid_retriever import HybridRetriever


def test_budget_parser_units():
    # 20k -> 0.2 Lakhs
    p1 = QueryParser.parse("properties under 20k")
    assert p1.max_price_lakhs == pytest.approx(0.2)

    # 20 thousand -> 0.2 Lakhs
    p2 = QueryParser.parse("under 20 thousand")
    assert p2.max_price_lakhs == pytest.approx(0.2)

    # 20000 -> 0.2 Lakhs
    p3 = QueryParser.parse("properties under 20000")
    assert p3.max_price_lakhs == pytest.approx(0.2)

    # 2 lakhs -> 2.0 Lakhs
    p4 = QueryParser.parse("apartments under 2 lakhs in Tirupati")
    assert p4.max_price_lakhs == pytest.approx(2.0)
    assert p4.city == "Tirupati"

    # 5 lakh -> 5.0 Lakhs
    p5 = QueryParser.parse("2 bedroom apartments under 5 lakh")
    assert p5.max_price_lakhs == pytest.approx(5.0)
    assert p5.bedrooms == 2

    # range 5 lakh to 20 lakh
    p6 = QueryParser.parse("houses between 5 lakh and 20 lakh")
    assert p6.min_price_lakhs == pytest.approx(5.0)
    assert p6.max_price_lakhs == pytest.approx(20.0)


def test_hard_budget_filtering_zero_matches():
    loader = DatasetLoader("data/properties.csv")
    properties = loader.load_data()
    docs = PropertyPreprocessor.process_all(properties)

    vec_store = VectorStoreManager("vectorstore_test_budget")
    vec_store.build_index(docs)
    retriever = HybridRetriever(vec_store, properties)

    # 1. properties under 20k
    _, m1 = retriever.search("properties under 20k")
    assert len(m1) == 0

    # 2. apartments under 20k
    _, m2 = retriever.search("apartments under 20k")
    assert len(m2) == 0

    # 3. apartments under 2 lakhs in Tirupati
    _, m3 = retriever.search("apartments under 2 lakhs in Tirupati")
    assert len(m3) == 0

    # 4. 2 bedroom apartments under 5 lakh
    _, m4 = retriever.search("2 bedroom apartments under 5 lakh")
    assert len(m4) == 0

    # 5. plots under 10 lakh in Tirupati
    _, m5 = retriever.search("plots under 10 lakh in Tirupati")
    assert len(m5) == 0

    # 6. houses between 5 lakh and 20 lakh
    _, m6 = retriever.search("houses between 5 lakh and 20 lakh")
    assert len(m6) == 0

    # 7. land under 15 lakh near Renigunta
    _, m7 = retriever.search("land under 15 lakh near Renigunta")
    assert len(m7) == 0


def test_hard_budget_filtering_valid_matches():
    loader = DatasetLoader("data/properties.csv")
    properties = loader.load_data()
    docs = PropertyPreprocessor.process_all(properties)

    vec_store = VectorStoreManager("vectorstore_test_valid")
    vec_store.build_index(docs)
    retriever = HybridRetriever(vec_store, properties)

    # Search apartments under 50 lakhs in Tirupati
    _, matches = retriever.search("apartments under 50 lakhs in Tirupati")
    assert len(matches) > 0
    for m in matches:
        assert m.property_item.price_lakhs <= 50.0
        assert m.property_item.city.lower() == "tirupati"
        assert m.property_item.property_type.lower() == "apartment"
