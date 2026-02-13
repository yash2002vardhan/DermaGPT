# DermaGPT

An AI-powered health and beauty product recommendation chatbot built with a Retrieval Augmented Generation (RAG) pipeline. DermaGPT helps users discover skincare, hair care, and vitamin/supplement products from a curated catalog of ~3,000 products, while also answering general wellness questions.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [RAG Pipeline Details](#rag-pipeline-details)
- [Supported Brands](#supported-brands)
- [Docker Deployment](#docker-deployment)
- [Limitations](#limitations)

## Features

- **Product Recommendations** - Find skincare, hair care, and vitamin/supplement products with natural language queries
- **Smart Filtering** - Filter by price range, brand, rating, and product category using conversational queries
- **Dual Vector Store** - FAISS (local) + Pinecone (cloud) for robust, redundant retrieval
- **General Wellness Q&A** - Answers health and beauty questions using web search (BraveSearch) or LLM knowledge
- **Conversation Memory** - Maintains chat context for follow-up questions within a session
- **Graceful Fallbacks** - Every component has a fallback path (see [Fallback Strategy](#fallback-strategy))

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Classification │  ← LLM-based (GPT-4o) with keyword fallback
│  "product" / "general"│
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐  ┌────────────┐
│Product │  │  General    │
│Pipeline│  │  Pipeline   │
└───┬────┘  └─────┬──────┘
    │             │
    ▼             ▼
┌────────────┐  ┌──────────────┐
│Category    │  │BraveSearch   │
│Detection   │  │  or          │
│+ Filter    │  │LLM Knowledge │
│Extraction  │  └──────┬───────┘
└─────┬──────┘         │
      ▼                │
┌──────────────────┐   │
│ Dual Retrieval   │   │
│ FAISS + Pinecone │   │
│ (Ensemble)       │   │
└─────┬────────────┘   │
      ▼                ▼
┌──────────────────────────┐
│   GPT-4o Response Gen    │
│   + Conversation Memory  │
└──────────────────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM | OpenAI GPT-4o | Query classification, response generation |
| Embeddings (primary) | OpenAI `text-embedding-3-small` (1536 dim) | Document & query vectorization |
| Embeddings (fallback) | `sentence-transformers/all-MiniLM-L6-v2` (384 dim) | Free local alternative |
| Vector Store (local) | FAISS | Local similarity search |
| Vector Store (cloud) | Pinecone (serverless, AWS us-east-1) | Cloud-based vector DB |
| RAG Framework | LangChain 0.3.20 | Orchestration, retrievers, chains |
| Web Search | BraveSearch API | General query augmentation |
| Frontend | Streamlit | Chat-based web UI |
| Observability | LangSmith | LangChain tracing & debugging |
| Containerization | Docker + Docker Compose | Deployment |

## Project Structure

```
DermaGPT/
├── main.py                 # Streamlit app + full RAG pipeline (production)
├── main.ipynb              # Development notebook (data preprocessing + RAG prototyping)
├── requirements.txt        # Python dependencies (145 packages)
├── Dockerfile              # Python 3.13-slim container
├── docker-compose.yml      # Service orchestration
├── .env                    # API keys (not committed)
├── .gitignore
├── assignment_details.pdf  # Project requirements spec
│
├── data.csv                # Original dataset (~3,000 products, 12 MB)
├── df_skin.csv             # Preprocessed skincare products (2,079 rows)
├── df_hair.csv             # Preprocessed hair care products (599 rows)
├── df_vits_supp.csv        # Preprocessed vitamins/supplements (302 rows)
│
└── faiss_index/            # FAISS vector store (auto-generated, gitignored)
```

## Getting Started

### Prerequisites

- Python 3.13+
- An OpenAI API key (required)
- A Pinecone API key (optional, falls back to FAISS-only)
- A BraveSearch API key (optional, falls back to LLM knowledge)

### Local Setup

```bash
# Clone the repository
git clone git@github.com:yash2002vardhan/DermaGPT.git
cd DermaGPT

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (see Environment Variables section)

# Run the app
streamlit run main.py
```

The app will be available at **http://localhost:8501**.

### Docker Setup

```bash
# Build and run
docker compose up --build

# Access at http://localhost:8501
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...          # Required - powers GPT-4o and embeddings
PINECONE_API_KEY=pc-...        # Optional - enables cloud vector store
BRAVE_API_KEY=BSA...           # Optional - enables web search for general queries
```

| Variable | Required | Impact if Missing |
|----------|----------|-------------------|
| `OPENAI_API_KEY` | Yes | App will not function |
| `PINECONE_API_KEY` | No | Falls back to FAISS-only retrieval |
| `BRAVE_API_KEY` | No | General queries use LLM knowledge instead of web search |

## Usage

### Product Queries

Ask about products using natural language. The system extracts category, price, brand, and rating filters automatically.

```
"Find me a sunscreen under 500 rupees"
"Hair shampoo for hairfall between 300-800 rs"
"Abbott brand skincare products"
"Vitamin B complex rated above 4 stars"
"Best moisturizer for acne-prone skin"
```

### General Wellness Queries

```
"How to treat acne naturally?"
"Why do I have dandruff?"
"What vitamins help with hair growth?"
"Tips for glowing skin"
```

The chatbot maintains conversation context, so follow-up questions like "show me cheaper options" or "any alternatives?" work as expected.

## Data Pipeline

### Source Data

The original dataset (`data.csv`) contains ~3,000 health and beauty products with fields including product title, vendor, category, price (INR), brand, description, FAQs, ingredients, key benefits, concerns, and review ratings.

### Preprocessing (main.ipynb)

1. **Load & Clean** - Rename columns, drop fields with >50% missing values
2. **Extract Metadata** - Parse structured fields (FAQs from JSON, benefits lists, ingredient lists, concern tags)
3. **Categorize** - Split into three category-specific CSVs:
   - `df_skin.csv` - 2,079 skincare products
   - `df_hair.csv` - 599 hair care products
   - `df_vits_supp.csv` - 302 vitamins & supplements
4. **Chunk & Embed** - Text splitting (500 char chunks, 100 char overlap) followed by vectorization

## RAG Pipeline Details

### 1. Query Classification

Incoming queries are classified as `"product"` or `"general"` using GPT-4o. If the LLM classification fails, a keyword-based fallback triggers on terms like: *buy, recommend, price, brand, product, suggest, sunscreen, shampoo, serum*, etc.

### 2. Category Detection

For product queries, regex patterns detect the relevant category:

- **Skin**: acne, pimple, moisturizer, sunscreen, wrinkles, dark spots, etc.
- **Hair**: hairfall, dandruff, shampoo, conditioner, split ends, frizz, etc.
- **Vitamins**: biotin, collagen, multivitamin, omega, zinc, iron, etc.

### 3. Filter Extraction

Natural language filters are parsed via regex:

- **Price**: `"under 500 rupees"`, `"between 300-800 rs"`, `"less than 1000"`
- **Brand**: Matched against 300+ indexed brands
- **Rating**: `"rated above 4"`, `"4.5 stars"`

### 4. Dual Vector Store Retrieval

An **Ensemble Retriever** combines results from both stores with equal weighting:

| Store | Type | Index | Dimensions |
|-------|------|-------|------------|
| FAISS | Local | `faiss_index/` directory | 1536 (OpenAI) or 384 (HuggingFace) |
| Pinecone | Cloud (serverless) | `clinikally-rag-2` | 1536 |

Pinecone uses three namespaces: `skin`, `hair`, `vitamins_supplements`. Both stores are populated from the preprocessed CSVs on first initialization.

### 5. Response Generation

- **Product queries**: Top 8 documents are formatted with title, price (INR), description, and category, then passed to GPT-4o for a numbered recommendation list
- **General queries**: BraveSearch fetches 5 web results (or LLM knowledge as fallback), combined with a health disclaimer

### 6. Conversation Memory

`ConversationBufferMemory` (LangChain) stores the full chat history per session, enabling contextual follow-ups. Memory is session-scoped via Streamlit's `st.session_state`.

### Fallback Strategy

| Component | Primary | Fallback |
|-----------|---------|----------|
| Query Classification | GPT-4o LLM call | Keyword pattern matching |
| Vector Store | Pinecone (cloud) | FAISS (local) |
| Retrieval | Ensemble (FAISS + Pinecone) | Individual retrievers |
| Web Search | BraveSearch API | LLM general knowledge |
| Embeddings | OpenAI `text-embedding-3-small` | HuggingFace `all-MiniLM-L6-v2` |

## Supported Brands

The system indexes 300+ brands for brand-specific filtering, including:

- **Indian Pharma**: Abbott, Cipla, Sun Pharma, Intas, Lupin, Glenmark
- **International Dermatology**: Galderma, La Roche-Posay, Bioderma, ISDIN, Sesderma
- **Consumer Skincare**: Neutrogena, Aveeno, CeraVe, Cetaphil, The Ordinary
- **K-Beauty**: COSRX, Some By Mi, Belif, The Face Shop, KAHI

See the full brand list in the `select_retrievers()` function in `main.py`.

## Docker Deployment

### Dockerfile

- Base image: `python:3.13-slim`
- Installs `build-essential` for compiled dependencies
- Exposes port `8501` (Streamlit default)
- Runs `streamlit run main.py --server.address=0.0.0.0`

### docker-compose.yml

- Mounts project directory as volume (enables hot reload during development)
- Loads environment variables from `.env`
- Restart policy: `unless-stopped`

```bash
# Start
docker compose up --build -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

## Limitations

- **Product catalog is static** - ~3,000 products with potentially outdated pricing; no live inventory integration
- **Three categories only** - Skincare, hair care, and vitamins/supplements
- **Session-only memory** - Conversation history is not persisted across sessions (no database backend)
- **No authentication** - No user accounts or personalized recommendations
- **Hardcoded model** - GPT-4o with non-configurable temperature/parameters
- **Max 8 products per query** - Response caps at 8 recommendations
- **OpenAI dependency** - Core functionality requires an OpenAI API key (paid)
