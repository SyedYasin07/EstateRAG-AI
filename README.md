# AI-Powered Real Estate Property Search and Recommendation System using RAG

An enterprise-grade, academic case-study implementation of an AI-powered Real Estate Property Search, Recommendation, and Comparison System using Retrieval-Augmented Generation (RAG).

The system enables users to express property search requirements in natural language (e.g. *"Find 2 BHK properties under ₹70 lakhs in Bangalore with parking"*), extracts structured search criteria alongside 384-dimensional vector embeddings, and generates grounded responses with hallucination-resistant constraints using retrieved listings.

---

## 🏗️ System Architecture

```text
               ┌──────────────────────┐
               │  Property CSV Data   │
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │ Dataset Loader       │
               │ & Schema Cleaner     │
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │ Document Preparation │
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │ SentenceTransformer  │
               │  all-MiniLM-L6-v2    │
               └──────────┬───────────┘
                          ↓
               ┌──────────────────────┐
               │   FAISS Vector DB    │
               └──────────┬───────────┘
                          │
 User Query               │
     ↓                    │
 ┌──────────────┐         │
 │ Query Parser │         │
 └──────┬───────┘         │
        ↓                 ↓
 ┌──────────────┐ ┌──────────────┐
 │   Metadata   │ │   Semantic   │
 │   Filtering  │ │ Vector Search│
 └──────┬───────┘ └──────┬───────┘
        └────────┬───────┘
                 ↓
      ┌──────────────────────┐
      │ Weighted Hybrid Rank │
      │ (Meta% + Vector%)    │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────┐
      │  Top-K Properties    │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────┐
      │ Context Builder      │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────┐
      │ Gemini / Groq LLM    │
      │ Strict Grounding     │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────┐
      │ Grounded Answer      │
      └──────────┬───────────┘
                 ↓
      ┌──────────────────────┐
      │ Streamlit UI Platform│
      └──────────────────────┘
```

---

## 🎯 Key Features

1. **Natural Language Search**: Translates unstructured conversational user prompts into precise multi-constraint queries.
2. **Weighted Hybrid Retrieval Engine**: Combines boolean pandas metadata filtering (budget limit, BHK count, sq.ft, location, property type, amenities) with FAISS semantic vector search.
3. **Strict Grounding & Anti-Hallucination**: Enforces system prompts instructing LLMs to cite verified `PROP-XXXX` listings and output explicit zero-match notifications when non-existent properties are requested.
4. **Interactive Enterprise Dashboard**: Displays market metrics (inventory size, average pricing, max area) alongside Plotly scatter charts and box plots.
5. **Property Comparison Matrix**: Enables side-by-side comparison of 2–3 listings with AI-generated comparative pros and cons summaries.
6. **Lifestyle Recommendation Engine**: Matches complex user lifestyle prompts (*"quiet family villa with garden and pool"*) to the best property options.
7. **Admin Management Panel**: Full CRUD control to view, add, edit, or delete property listings, with automatic FAISS index rebuilding.
8. **Offline Fallback Engine**: Works 100% out of the box locally even without external LLM API keys.

---

## 🛠️ Technology Stack

- **Core Language**: Python 3.10+
- **Data & Processing**: Pandas, Pydantic v2
- **Vector Search & Embeddings**: FAISS (`faiss-cpu`), SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM Integrations**: Google Gemini API (`google-genai`), Groq API, with deterministic offline synthesizer fallback
- **User Interface**: Streamlit with custom CSS themes & Plotly charts
- **Testing & Verification**: Pytest test suite

---

## 📊 Dataset Schema

The primary data source is `data/properties.csv` containing 60 listings across Indian tech hubs (Bangalore, Mumbai, Hyderabad, Pune, Delhi NCR, Chennai):

| Column Name | Type | Description |
| --- | --- | --- |
| `property_id` | string | Unique listing ID (e.g. `PROP-1001`) |
| `title` | string | Short descriptive property title |
| `location` | string | Neighborhood / locality |
| `city` | string | Metro city name |
| `price_lakhs` | float | Price in INR Lakhs (e.g., 68.5 = ₹68.5 Lakhs) |
| `area_sqft` | int | Carpet area in square feet |
| `bedrooms` | int | Number of bedrooms (BHK) |
| `bathrooms` | int | Number of bathrooms |
| `property_type` | string | Apartment, Villa, Independent House, Penthouse |
| `amenities` | list | Comma-separated amenities list |
| `furnishing` | string | Furnished, Semi-Furnished, Unfurnished |
| `age_years` | int | Age of building in years |
| `description` | string | Natural language lifestyle description |

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/real-estate-rag.git
cd real-estate-rag

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables (Optional)
Copy `.env.example` to `.env` and fill in your API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If omitted, the platform will seamlessly use the built-in Grounded Offline Synthesizer).*

### 4. Build FAISS Vector Index
```bash
python scripts/ingest_data.py
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## 🧪 Testing & Verification

Run the automated test suite covering dataset processing, metadata filtering, hybrid search, and anti-hallucination grounding:

```bash
pytest -v tests/
```

To execute the 10 sample test queries via CLI:
```bash
python scripts/test_queries.py
```

---

## 🧪 Sample Verification Queries

1. `"Find 2 BHK properties under ₹70 lakhs."` -> Verifies budget <= 70L, BHK=2 filtering.
2. `"Show properties larger than 1200 sq.ft."` -> Verifies sq.ft > 1200 filtering.
3. `"Find affordable properties in Bangalore."` -> Verifies city filtering & budget ranking.
4. `"Which property has the lowest price?"` -> Verifies sort order ranking.
5. `"Compare the top two properties."` -> Verifies comparison matrix.
6. `"Find a spacious family-friendly property with parking."` -> Verifies semantic vector search.
7. `"Show properties with 3 bedrooms."` -> Verifies BHK=3 filter.
8. `"Which property has the largest area?"` -> Verifies area sort ordering.
9. `"Find properties matching my budget and location."` -> Verifies multi-constraint search.
10. `"Show me properties that do not exist in the dataset."` -> **Demonstrates zero-hallucination grounding by returning "No properties in the available dataset satisfy all the specified requirements."**

---

## ⚠️ Limitations & Future Scope

### Current Limitations
- Static dataset source (`data/properties.csv`).
- Synthetic local vector store index.
- Regex + rule-based query parser for numerical extraction.

### Future Scope
- Live API integration with real-estate portals.
- Interactive map view with geocoding & neighborhood safety index.
- Multi-lingual query understanding (Hindi, Tamil, Kannada, Telugu).
- Voice search capabilities.
