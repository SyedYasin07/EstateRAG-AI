"""
Data preprocessor module for constructing rich embedding documents and metadata schemas.
"""

from typing import List, Dict, Any
from app.data.schema import PropertyItem


class Document:
    """Represents a preprocessed property document for RAG vector storage."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"<Document id={self.metadata.get('property_id')} content_len={len(self.page_content)}>"


class PropertyPreprocessor:
    """Transforms raw PropertyItem models into searchable documents and filterable metadata."""

    @staticmethod
    def create_document(property_item: PropertyItem) -> Document:
        """Converts a PropertyItem into a rich textual document optimized for semantic search."""
        amenities_str = ", ".join(property_item.amenities) if property_item.amenities else "None"
        
        loc_details = ", ".join([f for f in [property_item.locality or property_item.location, property_item.area, property_item.district, property_item.city, property_item.state, property_item.pincode] if f])
        
        # Build comprehensive property document text
        page_content = (
            f"Property ID: {property_item.property_id}\n"
            f"Title: {property_item.title}\n"
            f"Location: {loc_details}\n"
            f"City: {property_item.city}\n"
            f"State: {property_item.state}\n"
            f"District: {property_item.district}\n"
            f"Landmark: {property_item.landmark}\n"
            f"Property Type: {property_item.property_type}\n"
            f"Bedrooms: {property_item.bedrooms} BHK\n"
            f"Bathrooms: {property_item.bathrooms}\n"
            f"Price: {property_item.formatted_price} ({property_item.price_lakhs} Lakhs)\n"
            f"Carpet Area: {property_item.area_sqft} sq.ft (Rate: ₹{property_item.price_per_sqft}/sq.ft)\n"
            f"Furnishing: {property_item.furnishing}\n"
            f"Age of Property: {property_item.age_years} years\n"
            f"Amenities: {amenities_str}\n"
            f"Description: {property_item.description}"
        )

        metadata = {
            "property_id": property_item.property_id,
            "title": property_item.title,
            "location": property_item.location,
            "city": property_item.city,
            "state": property_item.state,
            "district": property_item.district,
            "locality": property_item.locality,
            "area": property_item.area,
            "price_lakhs": property_item.price_lakhs,
            "area_sqft": property_item.area_sqft,
            "bedrooms": property_item.bedrooms,
            "bathrooms": property_item.bathrooms,
            "property_type": property_item.property_type,
            "amenities": property_item.amenities,
            "furnishing": property_item.furnishing,
            "age_years": property_item.age_years,
            "image_urls": property_item.image_urls,
        }

        return Document(page_content=page_content, metadata=metadata)

    @classmethod
    def process_all(cls, properties: List[PropertyItem]) -> List[Document]:
        """Processes a list of PropertyItems into vector search documents."""
        return [cls.create_document(p) for p in properties]
