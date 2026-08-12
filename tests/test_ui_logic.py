"""
Tests for Property Comparison logic and AI analysis.
"""

import pytest
from app.data.dataset_loader import DatasetLoader
from app.services.rag_chain import RAGChain


@pytest.fixture(scope="module")
def properties():
    loader = DatasetLoader(data_path="data/properties.csv")
    return loader.load_data()


def test_comparison_two_properties(properties):
    # Find PROP-1005 and PROP-1008 if present, or any 2 properties
    prop1 = properties[0]
    prop2 = properties[1]
    
    comp_result = RAGChain.generate_comparison([prop1, prop2])
    
    assert comp_result is not None
    assert prop1.property_id in comp_result
    assert prop2.property_id in comp_result
    assert "Comparative Takeaways" in comp_result or "PROP-" in comp_result


def test_comparison_three_properties(properties):
    selected = properties[:3]
    comp_result = RAGChain.generate_comparison(selected)
    
    assert comp_result is not None
    for p in selected:
        assert p.property_id in comp_result


def test_property_image_safe_resolution(properties):
    from app.utils.image_config import get_property_image, DEFAULT_PLACEHOLDER_IMAGE
    
    p = properties[0]
    img = get_property_image(p)
    assert isinstance(img, str)
    assert len(img) > 10
    
    # Test None input
    assert get_property_image(None) == DEFAULT_PLACEHOLDER_IMAGE
