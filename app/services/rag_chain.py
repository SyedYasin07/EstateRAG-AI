"""
RAG Chain service for context construction, grounding enforcement, and LLM response generation.
"""

from typing import List, Dict, Any, Tuple, Optional
from app.data.schema import PropertyItem
from app.retrieval.hybrid_retriever import PropertyMatch
from app.retrieval.query_parser import ParsedQuery
from app.services.llm_factory import LLMFactory


GROUNDED_SYSTEM_PROMPT = """You are an expert AI Real Estate Search & Recommendation Assistant.
Your goal is to provide accurate, grounded property recommendations based ONLY on the provided context of retrieved property listings.

STRICT OPERATIONAL RULES:
1. Grounding & Anti-Hallucination: Rely EXCLUSIVELY on the provided Property Context. NEVER invent or assume property IDs, prices, locations, bedroom counts, or amenities that do not appear in the context.
2. Citation: Always cite the exact Property ID (e.g. PROP-1001, PROP-1004) when describing or recommending any listing.
3. Zero Matches: If the provided context is empty or contains no matching properties (or if the user requests non-existent properties), state clearly: "No properties in the available dataset satisfy all the specified requirements."
4. Distinction: Clearly distinguish verified facts in the listing from general purchasing advice.
5. Currency Format: Represent prices in Indian Rupees (e.g. ₹68.5 Lakhs or ₹2.40 Cr) exactly as listed in the context.
"""


class RAGChain:
    """RAG Service orchestrating context construction, strict system prompt enforcement, and grounded answers."""

    @staticmethod
    def build_context(matches: List[PropertyMatch]) -> str:
        """Formats retrieved PropertyMatch items into structured context for the LLM."""
        if not matches:
            return "NO MATCHING PROPERTIES FOUND IN DATASET."

        context_blocks = []
        for idx, match in enumerate(matches, 1):
            p = match.property_item
            amenities_str = ", ".join(p.amenities) if p.amenities else "None"
            block = (
                f"--- PROPERTY RECORD #{idx} ---\n"
                f"Property ID: {p.property_id}\n"
                f"Title: {p.title}\n"
                f"Location: {p.location}, {p.city}\n"
                f"Property Type: {p.property_type}\n"
                f"Bedrooms: {p.bedrooms} BHK | Bathrooms: {p.bathrooms}\n"
                f"Carpet Area: {p.area_sqft} sq.ft\n"
                f"Listed Price: {p.formatted_price} ({p.price_lakhs} Lakhs)\n"
                f"Price/sqft: ₹{p.price_per_sqft}/sq.ft\n"
                f"Furnishing: {p.furnishing} | Age: {p.age_years} years\n"
                f"Amenities: {amenities_str}\n"
                f"Match Score: {match.combined_score * 100:.1f}%\n"
                f"Key Highlights: {', '.join(match.match_reasons[:3])}\n"
                f"Description: {p.description}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    @classmethod
    def generate_response(
        self,
        user_query: str,
        parsed_query: ParsedQuery,
        matches: List[PropertyMatch],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generates grounded RAG response along with context metadata."""

        # Handle zero match or impossible query explicitly
        if not matches or parsed_query.is_impossible_query:
            if parsed_query.city or parsed_query.location:
                zero_match_msg = "No matching properties were found for this location in the available listings."
            else:
                zero_match_msg = "No properties in the available dataset satisfy all the specified requirements."
            return {
                "answer": zero_match_msg,
                "matches": [],
                "context": "No matching properties found.",
                "used_llm": False,
            }

        context_str = self.build_context(matches)

        # Build user prompt with conversation history context
        history_str = ""
        if conversation_history:
            history_blocks = [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in conversation_history[-2:]]
            history_str = "Prior Conversation History:\n" + "\n".join(history_blocks) + "\n\n"

        user_prompt = (
            f"{history_str}"
            f"User Query: \"{user_query}\"\n\n"
            f"Retrieved Property Context:\n{context_str}\n\n"
            f"Instructions: Provide a detailed, professional response summarizing the matching properties. "
            f"Mention the specific Property ID for each match and explain why it fits the query."
        )

        # 1. Attempt LLM API call
        llm_response = LLMFactory.get_llm_response(
            prompt=user_prompt, system_instruction=GROUNDED_SYSTEM_PROMPT
        )

        if llm_response:
            return {
                "answer": llm_response,
                "matches": matches,
                "context": context_str,
                "used_llm": True,
            }

        # 2. Offline Grounded Synthesizer Fallback
        fallback_answer = self._synthesize_offline_response(user_query, parsed_query, matches)
        return {
            "answer": fallback_answer,
            "matches": matches,
            "context": context_str,
            "used_llm": False,
        }

    @classmethod
    def generate_comparison(cls, selected_properties: List[PropertyItem]) -> str:
        """Generates grounded side-by-side comparative summary for selected properties."""
        if not selected_properties:
            return "No properties selected for comparison."

        matches = [
            PropertyMatch(
                property_item=p,
                combined_score=1.0,
                metadata_score=1.0,
                vector_score=1.0,
                match_reasons=["Selected for direct comparison"],
            )
            for p in selected_properties
        ]

        context_str = cls.build_context(matches)
        prompt = (
            f"Compare the following properties side-by-side based ONLY on their listing details:\n\n"
            f"{context_str}\n\n"
            f"Highlight key differences in Price, Area, Price/sq.ft, Bedroom configuration, Location, and Amenities. "
            f"Do not invent external info. Cite property IDs."
        )

        llm_resp = LLMFactory.get_llm_response(prompt, GROUNDED_SYSTEM_PROMPT)
        if llm_resp:
            return llm_resp

        # Offline Fallback for Comparison
        lines = [f"### Side-by-Side Comparison of {len(selected_properties)} Properties:\n"]
        for p in selected_properties:
            lines.append(
                f"- **{p.property_id} ({p.title})**: Price {p.formatted_price} | Area {p.area_sqft} sq.ft (₹{p.price_per_sqft}/sq.ft) | "
                f"Config {p.bedrooms} BHK, {p.bathrooms} Baths | Location {p.location}, {p.city} | Amenities: {', '.join(p.amenities[:4]) if p.amenities else 'None'}"
            )
        
        sorted_by_price = sorted(selected_properties, key=lambda x: x.price_lakhs)
        sorted_by_area = sorted(selected_properties, key=lambda x: x.area_sqft, reverse=True)
        
        lines.append(f"\n**Comparative Takeaways**:")
        lines.append(f"• **Most Affordable**: {sorted_by_price[0].property_id} listed at {sorted_by_price[0].formatted_price}.")
        lines.append(f"• **Largest Carpet Area**: {sorted_by_area[0].property_id} offering {sorted_by_area[0].area_sqft} sq.ft.")
        
        return "\n".join(lines)

    @staticmethod
    def _synthesize_offline_response(
        query: str, parsed: ParsedQuery, matches: List[PropertyMatch]
    ) -> str:
        """Synthesizes a grounded response using retrieved property records when offline."""
        count = len(matches)
        lines = [f"I found {count} property listing{'s' if count > 1 else ''} matching your search requirements:\n"]

        for idx, m in enumerate(matches, 1):
            p = m.property_item
            reasons_text = f" Matches criteria: {', '.join(m.match_reasons[:2])}." if m.match_reasons else ""
            lines.append(
                f"**{idx}. {p.title} ({p.property_id})**\n"
                f"- **Location**: {p.location}, {p.city}\n"
                f"- **Price**: {p.formatted_price} | **Area**: {p.area_sqft} sq.ft ({p.bedrooms} BHK, {p.bathrooms} Baths)\n"
                f"- **Highlights**: {p.property_type}, {p.furnishing}.{reasons_text}\n"
            )

        top_match = matches[0].property_item
        lines.append(
            f"\n**Top Recommendation**: **{top_match.property_id}** is the strongest match with an area of {top_match.area_sqft} sq.ft listed at {top_match.formatted_price}."
        )

        return "\n".join(lines)
