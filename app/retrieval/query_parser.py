"""
Query Parser module for parsing natural language real-estate queries into structured filter constraints.
"""

import re
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    raw_query: str
    max_price_lakhs: Optional[float] = None
    min_price_lakhs: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    min_area_sqft: Optional[int] = None
    max_area_sqft: Optional[int] = None
    city: Optional[str] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    furnishing: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    sort_order: Optional[str] = None
    is_impossible_query: bool = False


KNOWN_TYPES = {
    "apartment": "Apartment",
    "flat": "Apartment",
    "condo": "Apartment",
    "villa": "Villa",
    "house": "Independent House",
    "independent house": "Independent House",
    "builder floor": "Independent House",
    "penthouse": "Penthouse",
    "sky villa": "Penthouse",
    "land": "Land",
    "plot": "Land",
    "residential plot": "Land",
    "residential land": "Land",
}
KNOWN_AMENITIES = [
    "parking", "swimming pool", "pool", "gym", "elevator", "lift",
    "power backup", "security", "garden", "terrace", "clubhouse", "sea view"
]


class QueryParser:
    """Parses natural language property search queries into structured criteria."""

    @classmethod
    def parse(cls, query: str) -> ParsedQuery:
        q_lower = query.lower().strip()
        parsed = ParsedQuery(raw_query=query)

        # Check for explicit non-existent query pattern
        if any(phrase in q_lower for phrase in [
            "do not exist", "does not exist", "non-existent", "imaginary", "fake property", "mars", "moon"
        ]):
            parsed.is_impossible_query = True
            return parsed

        # 1. Extract Bedrooms (BHK)
        bhk_match = re.search(r"(\d+)\s*(?:bhk|bedroom|bed|br)", q_lower)
        if bhk_match:
            parsed.bedrooms = int(bhk_match.group(1))

        # 2. Extract Bathrooms
        bath_match = re.search(r"(\d+)\s*(?:bath|bathroom|baths)", q_lower)
        if bath_match:
            parsed.bathrooms = int(bath_match.group(1))

        # 3. Extract Budget (Lakhs & Crores)
        # e.g., under 70 lakhs, below 1.5 crore, budget 80l, under ₹70 lakhs, < 70 lakhs
        cr_under = re.search(r"(?:under|below|less than|within|budget of|<=|<|upto)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:cr|crore|crores)", q_lower)
        if cr_under:
            parsed.max_price_lakhs = float(cr_under.group(1)) * 100.0

        lakh_under = re.search(r"(?:under|below|less than|within|budget of|<=|<|upto)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lacs|lac|l)", q_lower)
        if lakh_under:
            parsed.max_price_lakhs = float(lakh_under.group(1))

        if not parsed.max_price_lakhs:
            # Check implicit "70 lakhs" or "₹70L"
            implicit_lakh = re.search(r"₹?\s*(\d+(?:\.\d+)?)\s*(?:lakhs|lakh|lacs)\b", q_lower)
            if implicit_lakh and ("under" in q_lower or "budget" in q_lower or "within" in q_lower or "below" in q_lower):
                parsed.max_price_lakhs = float(implicit_lakh.group(1))

        lakh_above = re.search(r"(?:above|more than|greater than|>=|>)\s*₹?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lacs|lac|l)", q_lower)
        if lakh_above:
            parsed.min_price_lakhs = float(lakh_above.group(1))

        # 4. Extract Area in Sq.Ft
        sqft_above = re.search(r"(?:larger than|more than|above|greater than|>|>=)\s*(\d+)\s*(?:sq\.?ft|sqft|square feet)", q_lower)
        if sqft_above:
            parsed.min_area_sqft = int(sqft_above.group(1))

        sqft_below = re.search(r"(?:smaller than|less than|under|below|<|<=)\s*(\d+)\s*(?:sq\.?ft|sqft|square feet)", q_lower)
        if sqft_below:
            parsed.max_area_sqft = int(sqft_below.group(1))

        # 5. Dynamic Location / City Extraction (matches "in <location>", "near <location>", "around <location>", "at <location>", or known city names)
        CITY_NORMALIZATION = {
            "tirupati": "Tirupati",
            "tirupathi": "Tirupati",
            "tiruphati": "Tirupati",
            "vijayawada": "Vijayawada",
            "bezawada": "Vijayawada",
            "guntur": "Guntur",
            "gunturu": "Guntur",
            "bangalore": "Bangalore",
            "bengaluru": "Bangalore",
            "hyderabad": "Hyderabad",
            "hyd": "Hyderabad",
            "mumbai": "Mumbai",
            "bombay": "Mumbai",
            "chennai": "Chennai",
            "madras": "Chennai",
            "pune": "Pune",
            "delhi": "Delhi NCR",
            "delhi ncr": "Delhi NCR",
            "gurgaon": "Delhi NCR",
            "noida": "Delhi NCR",
            "visakhapatnam": "Visakhapatnam",
            "vizag": "Visakhapatnam",
            "kurnool": "Kurnool",
        }

        # Check explicit known city match first
        for key, canonical_city in CITY_NORMALIZATION.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q_lower):
                parsed.city = canonical_city
                parsed.location = canonical_city
                break

        # If city not found via direct match, attempt dynamic regex extraction
        if not parsed.city:
            loc_match = re.search(r"\b(?:in|near|around|at)\s+([a-zA-Z\s\-]+)", query, re.IGNORECASE)
            if loc_match:
                raw_loc = loc_match.group(1).strip()
                words = raw_loc.split()
                clean_words = []
                stop_words = {
                    "under", "below", "above", "with", "having", "budget", "less", "more",
                    "larger", "smaller", "for", "properties", "property", "flats", "flat",
                    "homes", "home", "villas", "villa", "plots", "plot", "land", "house",
                    "lakhs", "lakh", "crores", "crore", "cr", "l"
                }
                for w in words:
                    if w.lower() in stop_words:
                        break
                    clean_words.append(w)
                if clean_words:
                    loc_name = " ".join(clean_words).strip()
                    loc_key = loc_name.lower()
                    normalized = CITY_NORMALIZATION.get(loc_key, loc_name.title())
                    parsed.city = normalized
                    parsed.location = normalized

        # 6. Extract Property Type
        for key, val in KNOWN_TYPES.items():
            if re.search(r"\b" + re.escape(key) + r"\b", q_lower):
                parsed.property_type = val
                break

        # 7. Extract Furnishing
        if "furnished" in q_lower and "semi" not in q_lower and "un" not in q_lower:
            parsed.furnishing = "Furnished"
        elif "semi-furnished" in q_lower or "semi furnished" in q_lower:
            parsed.furnishing = "Semi-Furnished"
        elif "unfurnished" in q_lower:
            parsed.furnishing = "Unfurnished"

        # 8. Extract Amenities
        for amenity in KNOWN_AMENITIES:
            if amenity in q_lower:
                clean_name = "Swimming Pool" if amenity in ["pool", "swimming pool"] else amenity.capitalize()
                if amenity in ["lift", "elevator"]:
                    clean_name = "Elevator"
                if clean_name not in parsed.amenities:
                    parsed.amenities.append(clean_name)

        # 9. Extract Sort Order / Special Intents
        if any(phrase in q_lower for phrase in ["lowest price", "cheapest", "least expensive", "minimum price"]):
            parsed.sort_order = "lowest_price"
        elif any(phrase in q_lower for phrase in ["highest price", "most expensive", "maximum price"]):
            parsed.sort_order = "highest_price"
        elif any(phrase in q_lower for phrase in ["largest area", "biggest", "maximum area", "largest sqft", "most spacious"]):
            parsed.sort_order = "largest_area"
        elif any(phrase in q_lower for phrase in ["smallest area", "smallest sqft"]):
            parsed.sort_order = "smallest_area"

        return parsed
