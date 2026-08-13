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

        # 3. Extract Budget (Lakhs & Crores, Thousands, K, Raw Numbers, Ranges)
        def amount_to_lakhs(val_str: str, unit_str: str = "") -> float:
            val = float(val_str.replace(",", ""))
            u = unit_str.lower().strip()
            if u in ["k", "thousand", "thousands"]:
                return (val * 1000.0) / 100000.0  # 20k -> 0.2 Lakhs
            elif u in ["cr", "crore", "crores"]:
                return val * 100.0  # 1.5 cr -> 150.0 Lakhs
            elif u in ["lakh", "lakhs", "lacs", "lac", "l"]:
                return val  # 2 lakh -> 2.0 Lakhs
            else:
                if val >= 1000.0:
                    return val / 100000.0  # 20000 -> 0.2 Lakhs
                else:
                    return val  # 50 -> 50.0 Lakhs

        # 3a. Check Ranges: "between X and Y", "X to Y", "X - Y"
        range_match = re.search(
            r"(?:between|from)?\s*(?:rs\.?\s*)?₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lacs|lac|l|cr|crore|crores)?\s*(?:and|to|-)\s*(?:rs\.?\s*)?₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lacs|lac|l|cr|crore|crores)?",
            q_lower
        )
        if range_match:
            v1_str, u1_str, v2_str, u2_str = (
                range_match.group(1),
                range_match.group(2) or "",
                range_match.group(3),
                range_match.group(4) or "",
            )
            if not u1_str and u2_str:
                u1_str = u2_str
            l1 = amount_to_lakhs(v1_str, u1_str)
            l2 = amount_to_lakhs(v2_str, u2_str)
            parsed.min_price_lakhs = min(l1, l2)
            parsed.max_price_lakhs = max(l1, l2)

        # 3b. Maximum Budget: "under X", "below X", "less than X", "within X", "budget of X", "upto X", "<= X", "< X"
        if parsed.max_price_lakhs is None:
            max_match = re.search(
                r"(?:under|below|less than|within|budget of|upto|<=|<|max budget of|maximum|budget)\s*(?:rs\.?\s*)?₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lacs|lac|l|cr|crore|crores)?",
                q_lower
            )
            if max_match:
                v_str, u_str = max_match.group(1), max_match.group(2) or ""
                parsed.max_price_lakhs = amount_to_lakhs(v_str, u_str)

        # 3c. Minimum Budget: "above X", "more than X", "greater than X", "over X", ">= X", "> X", "minimum"
        if parsed.min_price_lakhs is None:
            min_match = re.search(
                r"(?:above|more than|greater than|over|>=|>|min budget of|minimum)\s*(?:rs\.?\s*)?₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lacs|lac|l|cr|crore|crores)?",
                q_lower
            )
            if min_match:
                v_str, u_str = min_match.group(1), min_match.group(2) or ""
                parsed.min_price_lakhs = amount_to_lakhs(v_str, u_str)

        # 3d. Implicit standalone budget: e.g. "20k apartment", "₹20k flat", "2 lakh property", "50000 budget"
        if parsed.max_price_lakhs is None and parsed.min_price_lakhs is None:
            implicit_match = re.search(
                r"(?:rs\.?\s*)?₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(k|thousand|thousands|lakh|lakhs|lacs|lac|cr|crore|crores)\b",
                q_lower
            )
            if implicit_match:
                v_str, u_str = implicit_match.group(1), implicit_match.group(2)
                parsed.max_price_lakhs = amount_to_lakhs(v_str, u_str)

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
            if re.search(r"\b" + re.escape(key) + r"s?\b", q_lower):
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
