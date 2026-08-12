"""
Tests for Data Loader, Schema, and Preprocessor.
"""

import os
import pytest
from app.data.schema import PropertyItem
from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor


def test_property_item_schema():
    prop = PropertyItem(
        property_id="PROP-9999",
        title="Test Luxury Apartment",
        location="Koramangala",
        city="Bangalore",
        price_lakhs=120.0,
        area_sqft=1500,
        bedrooms=3,
        bathrooms=3,
        property_type="Apartment",
        amenities="Parking, Gym, Swimming Pool",
        furnishing="Furnished",
        age_years=2,
        description="A beautiful test apartment."
    )
    assert prop.property_id == "PROP-9999"
    assert prop.formatted_price == "₹1.20 Cr"
    assert prop.price_per_sqft == 8000.0
    assert "Parking" in prop.amenities
    assert "Gym" in prop.amenities
    assert len(prop.amenities) == 3


def test_dataset_loader():
    loader = DatasetLoader(data_path="data/properties.csv")
    properties = loader.load_data()
    assert len(properties) >= 80
    summary = loader.get_summary()
    assert summary.total_properties >= 80
    assert "Bangalore" in summary.cities
    assert summary.min_price_lakhs > 0


def test_preprocessor():
    prop = PropertyItem(
        property_id="PROP-1001",
        title="Test Flat",
        location="Indiranagar",
        city="Bangalore",
        price_lakhs=65.0,
        area_sqft=1100,
        bedrooms=2,
        bathrooms=2,
        property_type="Apartment",
        amenities=["Parking"],
        furnishing="Semi-Furnished",
        age_years=1,
        description="Nice test flat."
    )
    doc = PropertyPreprocessor.create_document(prop)
    assert doc.metadata["property_id"] == "PROP-1001"
    assert "65.0 Lakhs" in doc.page_content
    assert "Indiranagar" in doc.page_content
