"""
Tests for Query Parser, Location Independence, Image URL Support, and Hybrid Retriever.
"""

import os
import pandas as pd
import pytest
from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.data.schema import PropertyItem
from app.retrieval.query_parser import QueryParser
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain
from app.utils.image_config import DEFAULT_PLACEHOLDER_IMAGE, get_property_image


@pytest.fixture(scope="module")
def setup_retriever():
    loader = DatasetLoader(data_path="data/properties.csv")
    properties = loader.load_data()
    docs = PropertyPreprocessor.process_all(properties)
    
    vec_store = VectorStoreManager(vector_store_dir="vectorstore_test")
    vec_store.build_index(docs)
    
    retriever = HybridRetriever(vector_store=vec_store, properties=properties)
    return retriever, properties, vec_store


def test_query_parser_dynamic_location():
    parsed_tirupati = QueryParser.parse("Show properties in Tirupati")
    assert parsed_tirupati.city == "Tirupati"

    parsed_land = QueryParser.parse("Show land properties in Tirupati under ₹50 lakhs")
    assert parsed_land.city == "Tirupati"
    assert parsed_land.property_type == "Land"
    assert parsed_land.max_price_lakhs == 50.0

    parsed_guntur = QueryParser.parse("Show properties in Guntur")
    assert parsed_guntur.city == "Guntur"


def test_duplicate_property_ids_dataset():
    df = pd.read_csv("data/properties.csv")
    assert df["property_id"].is_unique == True
    duplicate_ids = df[df["property_id"].duplicated(keep=False)]
    assert len(duplicate_ids) == 0


def test_city_spelling_normalization():
    p1 = QueryParser.parse("Show properties in Tirupathi")
    assert p1.city == "Tirupati"

    p1b = QueryParser.parse("Show properties in Tiruphati")
    assert p1b.city == "Tirupati"

    p2 = QueryParser.parse("Properties in Vizag")
    assert p2.city == "Visakhapatnam"

    p3 = QueryParser.parse("3 BHK in Hyd")
    assert p3.city == "Hyderabad"

    p4 = QueryParser.parse("Flats in Bengaluru")
    assert p4.city == "Bangalore"

    p5 = QueryParser.parse("Plots in Bezawada")
    assert p5.city == "Vijayawada"


def test_image_resolver_crash_proof():
    # Test None / missing
    assert get_property_image(None) == DEFAULT_PLACEHOLDER_IMAGE
    
    # Test empty property gets safe non-empty fallback image
    prop_empty = PropertyItem(
        property_id="PROP-TEST",
        title="Test",
        location="Loc",
        city="Tirupati",
        price_lakhs=50.0,
        area_sqft=1000,
        bedrooms=2,
        bathrooms=2,
        property_type="Apartment",
        amenities=[],
        description="Desc",
        image_urls="",
    )
    fallback_img = get_property_image(prop_empty)
    assert isinstance(fallback_img, str) and len(fallback_img) > 10
    assert fallback_img.startswith(("http://", "https://"))

    # Test valid image
    prop_valid = PropertyItem(
        property_id="PROP-TEST2",
        title="Test 2",
        location="Loc",
        city="Tirupati",
        price_lakhs=50.0,
        area_sqft=1000,
        bedrooms=2,
        bathrooms=2,
        property_type="Apartment",
        amenities=[],
        description="Desc",
        image_urls="https://images.unsplash.com/photo-1560518883-ce09059eeffa",
    )
    assert get_property_image(prop_valid) == "https://images.unsplash.com/photo-1560518883-ce09059eeffa"


def test_retriever_tirupati_location(setup_retriever):
    retriever, _, _ = setup_retriever
    query = "Show properties in Tirupati"
    parsed, matches = retriever.search(query, top_k=10)
    
    assert len(matches) >= 5
    for match in matches:
        p = match.property_item
        assert "tirupati" in p.city.lower() or "tirupati" in p.location.lower()
        assert len(p.image_urls) > 0


def test_retriever_tirupati_land(setup_retriever):
    retriever, _, _ = setup_retriever
    query = "Show land properties in Tirupati"
    parsed, matches = retriever.search(query, top_k=10)
    
    assert len(matches) > 0
    for match in matches:
        p = match.property_item
        assert p.property_type == "Land"
        assert "tirupati" in p.city.lower()


def test_recommendation_search_tirupati(setup_retriever):
    retriever, _, _ = setup_retriever
    query = "I need a family home in Tirupati"
    parsed, matches = retriever.search(query, top_k=5)
    
    assert len(matches) > 0
    assert any("tirupati" in m.property_item.city.lower() for m in matches)


def test_retriever_non_existent_city(setup_retriever):
    retriever, _, _ = setup_retriever
    query = "Show me properties in Kurnool"
    parsed, matches = retriever.search(query, top_k=5)
    assert len(matches) == 0

    response = RAGChain.generate_response(query, parsed, matches)
    assert "No matching properties were found for this location" in response["answer"]
