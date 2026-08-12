"""
Automated script to run all 12 evaluation test queries through the hybrid retriever and RAG pipeline.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain

TEST_QUERIES = [
    "1. Show properties in Tirupati",
    "2. Show land properties in Tirupati",
    "3. 2 BHK properties in Tirupati",
    "4. 2 BHK under 70 lakhs in Tirupati",
    "5. Properties near Alipiri",
    "6. Residential plots above 2000 sq.ft in Tirupati",
    "7. 3 BHK properties in Vijayawada",
    "8. Properties in Hyderabad",
    "9. Show properties in Chennai",
    "10. Find properties that do not exist in the dataset",
    "11. Show properties in Guntur",
    "12. Find properties matching my budget and location",
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=== EstateRAG AI — Automated Evaluation Query Suite ===\n")
    loader = DatasetLoader(data_path="data/properties.csv")
    properties = loader.load_data()
    docs = PropertyPreprocessor.process_all(properties)

    vec_manager = VectorStoreManager(vector_store_dir="vectorstore")
    if not vec_manager.load_index():
        print("Building vector index...")
        vec_manager.build_index(docs)

    retriever = HybridRetriever(vector_store=vec_manager, properties=properties)

    for q in TEST_QUERIES:
        print(f"\n==================================================")
        print(f"QUERY: {q}")
        print(f"==================================================")

        parsed, matches = retriever.search(q, top_k=10)
        print(f"Parsed Criteria: BHK={parsed.bedrooms}, MaxBudget={parsed.max_price_lakhs}L, MinArea={parsed.min_area_sqft}sqft, City={parsed.city}, Location={parsed.location}, Amenities={parsed.amenities}, Sort={parsed.sort_order}")
        print(f"Retrieved Top Matches: {len(matches)}")

        for m in matches[:5]:
            p = m.property_item
            print(f"  -> [{p.property_id}] {p.title} | {p.locality or p.location}, {p.city} | {p.bedrooms}BHK | {p.area_sqft}sqft | {p.formatted_price} | Combined Score: {m.combined_score*100:.1f}%")

        response = RAGChain.generate_response(q, parsed, matches)
        print(f"\nAI GROUNDED RESPONSE:\n{response['answer']}\n")


if __name__ == "__main__":
    main()
