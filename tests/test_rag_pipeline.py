"""
Tests for RAG Chain and Grounded Output Generation.
"""

import pytest
from app.data.dataset_loader import DatasetLoader
from app.data.preprocessor import PropertyPreprocessor
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.rag_chain import RAGChain


@pytest.fixture(scope="module")
def setup_rag():
    loader = DatasetLoader(data_path="data/properties.csv")
    properties = loader.load_data()
    docs = PropertyPreprocessor.process_all(properties)
    vec_store = VectorStoreManager(vector_store_dir="vectorstore_test")
    vec_store.build_index(docs)
    retriever = HybridRetriever(vector_store=vec_store, properties=properties)
    return retriever


def test_rag_valid_query(setup_rag):
    retriever = setup_rag
    query = "Find 2 BHK properties under ₹70 lakhs in Bangalore"
    parsed, matches = retriever.search(query, top_k=3)
    response = RAGChain.generate_response(query, parsed, matches)
    
    assert response["answer"] is not None
    assert len(response["matches"]) > 0
    assert "PROP-" in response["answer"]


def test_rag_non_existent_property_query(setup_rag):
    retriever = setup_rag
    query = "Show me properties that do not exist in the dataset"
    parsed, matches = retriever.search(query, top_k=5)
    response = RAGChain.generate_response(query, parsed, matches)
    
    assert "No properties in the available dataset satisfy all the specified requirements." in response["answer"]
    assert len(response["matches"]) == 0
