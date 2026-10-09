# Semantic search and RAG

A small FastAPI backend and Streamlit frontend over the same company documents.
The two features are separate pipelines.

Semantic search embeds each document as one vector and returns the nearest documents with a cosine score. It does not write an answer.

RAG splits documents into overlapping chunks, retrieves the nearest chunks, builds a prompt that says to answer only from that context, and then generates an answer.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In one terminal, from `backend`:

```powershell
uvicorn --app-dir backend main:app --reload --port 8000
```

The first start downloads the local embedding model (`BAAI/bge-small-en-v1.5`).

In another terminal, from the project folder:

```powershell
streamlit run frontend/app.py
```

## Gemini

Copy `.env.example` to `.env` in the project folder and paste your key into `GEMINI_API_KEY`. RAG then calls `gemini-3.6-flash`.

Restart the API after saving the key. If the key is missing, RAG still retrieves chunks and quotes the best one, and the screen labels that mode `extractive_fallback`.

## Evaluation

The Streamlit page has a **Run evaluation** button. It scores 24 labeled questions at the three chunk presets and reports Precision@3, Precision@5, Recall@5, MRR, answer accuracy, and abstention. With a Gemini key this calls the model once per question per preset.
