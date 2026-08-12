"""
Script to ingest property dataset, clean records, and build FAISS vector index.
"""

import sys
import os

# Add root project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.retrieval.vector_store import VectorStoreManager


def main():
    data_path = os.environ.get("DATA_PATH", "data/properties.csv")
    vector_dir = os.environ.get("VECTOR_STORE_DIR", "vectorstore")

    print(f"=== Real Estate Property Ingestion Pipeline ===")
    print(f"[1/3] Loading dataset from: {data_path}")
    loader = DatasetLoader(data_path=data_path)
    properties = loader.load_data()
    print(f"  -> Successfully loaded {len(properties)} property records.")

    print(f"[2/3] Transforming properties into searchable documents...")
    documents = PropertyPreprocessor.process_all(properties)
    print(f"  -> Created {len(documents)} preprocessed documents.")

    print(f"[3/3] Generating vector embeddings and building FAISS index at '{vector_dir}'...")
    vec_manager = VectorStoreManager(vector_store_dir=vector_dir)
    count = vec_manager.build_index(documents)
    print(f"  -> Ingestion complete! {count} documents indexed successfully in FAISS.")

    summary = loader.get_summary()
    print(f"\n--- Dataset Summary Stats ---")
    print(f"Total Properties: {summary.total_properties}")
    print(f"Cities ({len(summary.cities)}): {', '.join(summary.cities)}")
    print(f"Property Types: {', '.join(summary.property_types)}")
    print(f"Price Range: Rs. {summary.min_price_lakhs}L - Rs. {summary.max_price_lakhs}L (Avg: Rs. {summary.avg_price_lakhs}L)")
    print(f"Area Range: {summary.min_area_sqft} - {summary.max_area_sqft} sq.ft")


if __name__ == "__main__":
    main()
