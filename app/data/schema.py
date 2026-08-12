"""
Property Pydantic Schema and Data Models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PropertyItem(BaseModel):
    property_id: str = Field(..., description="Unique property identifier, e.g., PROP-1001")
    title: str = Field(..., description="Property title")
    location: str = Field(..., description="Specific neighborhood/area name")
    city: str = Field(..., description="Major city name")
    price_lakhs: float = Field(..., description="Price in INR Lakhs (e.g. 70.5 means 70.5 Lakhs)")
    area_sqft: int = Field(..., description="Total carpet area in square feet")
    bedrooms: int = Field(..., description="Number of bedrooms (BHK)")
    bathrooms: int = Field(..., description="Number of bathrooms")
    property_type: str = Field(..., description="Type of property: Apartment, Villa, Independent House, Penthouse")
    amenities: List[str] = Field(default_factory=list, description="List of available amenities")
    furnishing: str = Field("Unfurnished", description="Furnishing status: Furnished, Semi-Furnished, Unfurnished")
    age_years: int = Field(0, description="Age of property in years")
    description: str = Field("", description="Detailed narrative description")
    state: str = Field("", description="State name, e.g. Andhra Pradesh, Karnataka")
    district: str = Field("", description="District name")
    locality: str = Field("", description="Locality / Neighborhood")
    area: str = Field("", description="Area name")
    landmark: str = Field("", description="Prominent landmark")
    pincode: str = Field("", description="Pincode / Zip Code")
    image_urls: List[str] = Field(default_factory=list, description="List of image URLs or relative asset paths")

    @field_validator("price_lakhs", "area_sqft", "bedrooms", "bathrooms", mode="before")
    @classmethod
    def parse_numeric(cls, v):
        if v is None or v == "":
            return 0
        if isinstance(v, str):
            # Clean commas or currency symbols
            clean_str = v.replace(",", "").replace("₹", "").replace("$", "").strip()
            return float(clean_str)
        return v

    @field_validator("amenities", "image_urls", mode="before")
    @classmethod
    def parse_list_fields(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @property
    def formatted_price(self) -> str:
        """Returns human readable price format."""
        if self.price_lakhs >= 100:
            crores = self.price_lakhs / 100.0
            return f"₹{crores:.2f} Cr"
        return f"₹{self.price_lakhs:.1f} Lakhs"

    @property
    def price_per_sqft(self) -> float:
        """Calculates price per sq ft in INR."""
        if self.area_sqft <= 0:
            return 0.0
        # price_lakhs * 100,000 / area_sqft
        return round((self.price_lakhs * 100000.0) / self.area_sqft, 2)


class DatasetSummary(BaseModel):
    total_properties: int
    cities: List[str]
    property_types: List[str]
    min_price_lakhs: float
    max_price_lakhs: float
    avg_price_lakhs: float
    min_area_sqft: int
    max_area_sqft: int
    available_amenities: List[str]
