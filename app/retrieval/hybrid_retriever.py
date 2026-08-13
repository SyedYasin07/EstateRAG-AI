"""
Hybrid Retriever combining Structured Metadata Filtering and FAISS Vector Similarity Search
using Weighted Hybrid Scoring.
"""

from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel
from app.data.schema import PropertyItem
from app.data.preprocessor import Document
from app.retrieval.query_parser import ParsedQuery, QueryParser
from app.retrieval.vector_store import VectorStoreManager


class PropertyMatch(BaseModel):
    property_item: PropertyItem
    combined_score: float
    metadata_score: float
    vector_score: float
    match_reasons: List[str]


class HybridRetriever:
    """Performs hybrid property search combining metadata filtering and FAISS vector similarity."""

    def __init__(self, vector_store: VectorStoreManager, properties: List[PropertyItem]):
        self.vector_store = vector_store
        self.properties = properties
        self.property_map: Dict[str, PropertyItem] = {p.property_id: p for p in properties}

    def _evaluate_metadata(self, item: PropertyItem, parsed: ParsedQuery) -> Tuple[float, List[str]]:
        """Calculates metadata compliance score (0.0 to 1.0) and list of match reasons."""
        checks = 0
        passed = 0
        reasons = []

        # 1. Bedrooms
        if parsed.bedrooms is not None:
            checks += 1
            if item.bedrooms == parsed.bedrooms:
                passed += 1
                reasons.append(f"Matches {parsed.bedrooms} BHK requirement")
            else:
                reasons.append(f"Differs in bedrooms ({item.bedrooms} BHK vs {parsed.bedrooms} requested)")

        # 2. Max Price / Budget
        if parsed.max_price_lakhs is not None:
            checks += 1
            if item.price_lakhs <= parsed.max_price_lakhs:
                passed += 1
                reasons.append(f"Within budget (₹{item.price_lakhs:.1f}L <= ₹{parsed.max_price_lakhs:.1f}L)")
            else:
                reasons.append(f"Exceeds budget (₹{item.price_lakhs:.1f}L > ₹{parsed.max_price_lakhs:.1f}L)")

        # 3. Min Price
        if parsed.min_price_lakhs is not None:
            checks += 1
            if item.price_lakhs >= parsed.min_price_lakhs:
                passed += 1
                reasons.append(f"Price above minimum limit (₹{item.price_lakhs:.1f}L)")

        # 4. Min Area (Sq.Ft)
        if parsed.min_area_sqft is not None:
            checks += 1
            if item.area_sqft >= parsed.min_area_sqft:
                passed += 1
                reasons.append(f"Spacious area ({item.area_sqft} sq.ft >= {parsed.min_area_sqft} sq.ft)")
            else:
                reasons.append(f"Smaller area than requested ({item.area_sqft} sq.ft)")

        # 5. Dynamic Location Match (matches city, state, district, location, locality, area, landmark)
        loc_target = (parsed.city or parsed.location or "").lower()
        if loc_target:
            checks += 1
            item_loc_fields = [
                item.city, item.state, item.district, item.location,
                item.locality, item.area, item.landmark
            ]
            loc_matched = any(loc_target in f.lower() for f in item_loc_fields if f)
            if loc_matched:
                passed += 1
                reasons.append(f"Located in {item.city or item.location}")
            else:
                reasons.append(f"Location mismatch ({loc_target} not found in listing)")

        # 6. Property Type
        if parsed.property_type:
            checks += 1
            if item.property_type.lower() == parsed.property_type.lower():
                passed += 1
                reasons.append(f"Property type matches {item.property_type}")
            else:
                reasons.append(f"Property type mismatch ({item.property_type} vs {parsed.property_type})")

        # 7. Amenities Match
        if parsed.amenities:
            for amenity in parsed.amenities:
                checks += 1
                item_amenities_lower = [a.lower() for a in item.amenities]
                if amenity.lower() in item_amenities_lower:
                    passed += 1
                    reasons.append(f"Includes amenity: {amenity}")

        if checks == 0:
            return 1.0, ["General query match"]

        score = float(passed / checks)
        return score, reasons

    def search(self, query_text: str, top_k: int = 5) -> Tuple[ParsedQuery, List[PropertyMatch]]:
        """Executes weighted hybrid search returning ranked property matches."""
        parsed_query = QueryParser.parse(query_text)

        # Handle impossible/out-of-bound non-existent queries
        if parsed_query.is_impossible_query:
            return parsed_query, []

        # 1. Perform FAISS Semantic Vector Retrieval
        vector_results = self.vector_store.similarity_search(query_text, top_k=len(self.properties))
        vector_score_map: Dict[str, float] = {}
        for doc, score in vector_results:
            prop_id = doc.metadata.get("property_id")
            if prop_id:
                vector_score_map[prop_id] = score

        # 2. Compute Hybrid Scores for all properties
        matches: List[PropertyMatch] = []
        loc_target = (parsed_query.city or parsed_query.location or "").lower()

        for prop in self.properties:
            meta_score, reasons = self._evaluate_metadata(prop, parsed_query)
            vec_score = vector_score_map.get(prop.property_id, 0.5)

            is_disqualified = False
            # Budget checks
            if parsed_query.max_price_lakhs is not None and prop.price_lakhs > parsed_query.max_price_lakhs:
                is_disqualified = True
            if parsed_query.min_price_lakhs is not None and prop.price_lakhs < parsed_query.min_price_lakhs:
                is_disqualified = True
            # Bedroom check
            if parsed_query.bedrooms is not None and prop.bedrooms != parsed_query.bedrooms:
                is_disqualified = True
            # Property type check
            if parsed_query.property_type and prop.property_type.lower() != parsed_query.property_type.lower():
                is_disqualified = True
            # Area check
            if parsed_query.min_area_sqft is not None and prop.area_sqft < parsed_query.min_area_sqft:
                is_disqualified = True
            if parsed_query.max_area_sqft is not None and prop.area_sqft > parsed_query.max_area_sqft:
                is_disqualified = True
            # Location check across all location fields
            if loc_target:
                prop_loc_fields = [
                    prop.city, prop.state, prop.district, prop.location,
                    prop.locality, prop.area, prop.landmark
                ]
                if not any(loc_target in f.lower() for f in prop_loc_fields if f):
                    is_disqualified = True

            if is_disqualified:
                meta_score = 0.0

            # Weighted Hybrid Score: 60% Metadata + 40% Vector Similarity
            combined_score = round(0.6 * meta_score + 0.4 * vec_score, 4)

            # Strict filter: Disqualified properties are strictly excluded if hard constraints fail
            if not is_disqualified:
                matches.append(
                    PropertyMatch(
                        property_item=prop,
                        combined_score=combined_score,
                        metadata_score=round(meta_score, 4),
                        vector_score=round(vec_score, 4),
                        match_reasons=reasons,
                    )
                )

        # 3. Handle Special Sort Intents (lowest price, largest area, etc.)
        if parsed_query.sort_order == "lowest_price":
            matches.sort(key=lambda m: m.property_item.price_lakhs)
        elif parsed_query.sort_order == "highest_price":
            matches.sort(key=lambda m: m.property_item.price_lakhs, reverse=True)
        elif parsed_query.sort_order == "largest_area":
            matches.sort(key=lambda m: m.property_item.area_sqft, reverse=True)
        elif parsed_query.sort_order == "smallest_area":
            matches.sort(key=lambda m: m.property_item.area_sqft)
        else:
            matches.sort(key=lambda m: m.combined_score, reverse=True)

        return parsed_query, matches[:top_k]
