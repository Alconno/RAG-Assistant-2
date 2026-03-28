# README

This project is a simple Retrieval-Augmented Generation (RAG) system
that ingests files (PDFs, images, etc.), converts them to text, stores
them in a vector database, and enables querying through an LLM.

## Setup

``` bash
git clone <repo>
cd <repo>
```

Install dependencies (use separate environments):

**FastAPI (model hosting):**

``` bash
pip install -r requirements-fastapi.txt
```

**Streamlit (UI + Weaviate ops):**

``` bash
pip install -r requirements-streamlit.txt
```

## Finetune Model

``` bash
python .\models\qwen06b\finetune.py
```

## Run

**1. Host models (FastAPI):**

``` bash
python .\fast_api\host_models.py
```

**2. Start Streamlit app:**

``` bash
python -m streamlit run main.py
```

### Examples
- Examples you can test with are in the examples folder