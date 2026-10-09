"""Streamlit UI for the two pipelines.

Start the API first, from the backend folder:

    uvicorn main:app --reload --port 8000

Then, from the project folder:

    streamlit run frontend/app.py

The same question is sent to both endpoints. Semantic search shows the
nearest documents and their scores. RAG shows an answer written from
retrieved chunks, plus the chunks and the prompt that produced it.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Short labels for the buttons. The value is the full question sent to the API.
SAMPLES = [
    ("Vacation days", "How many vacation days do employees get?"),
    ("Official trip", "How do I get money back for my official trip?"),
    ("Student loan", "What expenses can a student loan cover?"),
    ("17.4% NIM", "17.4% NIM"),
    ("Stock price", "What is the company's stock price?"),
]

# The three chunk settings from the chunking experiment.
CHUNK_PRESETS = {
    "A — size 200, overlap 20": (200, 20),
    "B — size 500, overlap 50": (500, 50),
    "C — size 1000, overlap 100": (1000, 100),
}


st.set_page_config(page_title="Semantic search and RAG", layout="wide")
st.title("Northwind Financial knowledge demo")
st.caption(
    "Semantic search stops at the closest documents. "
    "RAG retrieves chunks, puts them in a prompt, and answers from that context."
)


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def show_hit(hit: dict) -> None:
    """One retrieved row: where it came from, how close it was, and the text."""
    score = hit["score"]
    location = hit["title"]
    if hit.get("chunk_index") is not None:
        location = f"{location} · chunk {hit['chunk_index']}"
    st.markdown(f"**{hit['rank']}. {location}**")
    st.caption(f"{hit['source']} · {hit['section']} · page {hit['page']} · cosine {score:.3f}")
    # The bar is only a visual for a score that already sits roughly in 0..1.
    st.progress(min(max(score, 0.0), 1.0))
    st.write(hit["text"])


with st.sidebar:
    st.header("How the two calls differ")
    st.markdown(
        """
**Semantic search** embeds each full document and returns the nearest
ones. There is no written answer. A question the corpus does not cover
still gets the nearest documents.

**RAG** splits each document into overlapping chunks, retrieves a few
chunks, and asks a model to answer using only those chunks.

Both use the same embedding model. They do not use the same index.
        """
    )
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(
            f"API ready · {health['documents']} documents · {health['embedding_model']}"
        )
    except requests.RequestException:
        st.error("API is not running. Start it with uvicorn on port 8000.")


def _use_sample(text: str) -> None:
    # Runs before the text input is created, which is the only time
    # Streamlit allows this widget's value to be replaced.
    st.session_state.question = text


if "question" not in st.session_state:
    st.session_state.question = SAMPLES[0][1]

st.text_input("Question", key="question")
button_row = st.columns(len(SAMPLES))
for column, (label, question) in zip(button_row, SAMPLES):
    column.button(
        label,
        use_container_width=True,
        on_click=_use_sample,
        args=(question,),
    )

controls = st.columns([2, 1, 1])
with controls[0]:
    # Preset B is the middle setting and the one the API builds at startup.
    preset_name = st.selectbox("RAG chunk preset", list(CHUNK_PRESETS), index=1)
with controls[1]:
    search_k = st.slider("Semantic search top-k", 1, 8, 5)
with controls[2]:
    rag_k = st.slider("RAG top-k", 1, 8, 3)

chunk_size, chunk_overlap = CHUNK_PRESETS[preset_name]
ask = st.button("Ask both", type="primary")

if ask:
    question = st.session_state.question.strip()
    if not question:
        st.warning("Enter a question first.")
    else:
        search_column, rag_column = st.columns(2)

        with search_column:
            st.subheader("Semantic search")
            st.caption("Nearest whole documents. No generation.")
            try:
                # Retrieval only. top-k defaults to 5, matching a search results list.
                found = api_post("/search", {"query": question, "top_k": search_k})
            except requests.RequestException as exc:
                st.error(f"Search failed: {exc}")
            else:
                if found["low_confidence"]:
                    st.warning(
                        "The nearest documents are still shown. "
                        f"The best score is under {found['threshold']:.2f}, "
                        "so none of them is a confident match. "
                        "Vector search does not know how to say it has no answer."
                    )
                for hit in found["results"]:
                    show_hit(hit)
                    st.divider()

        with rag_column:
            st.subheader("RAG")
            st.caption(
                f"Chunks of {chunk_size} characters, overlap {chunk_overlap}. "
                "Then a grounded answer."
            )
            try:
                # Smaller top-k than search: these chunks become the model context.
                answered = api_post(
                    "/rag",
                    {
                        "query": question,
                        "top_k": rag_k,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                )
            except requests.RequestException as exc:
                st.error(f"RAG failed: {exc}")
            else:
                st.markdown("**Answer**")
                st.write(answered["answer"])
                st.caption(f"Generation: {answered['generation_mode']}")
                if answered["refused"]:
                    st.info(
                        "The answer was refused because the best chunk was below "
                        "the similarity threshold. The sources are what retrieval found anyway."
                    )
                with st.expander("Retrieved chunks"):
                    for hit in answered["sources"]:
                        show_hit(hit)
                        st.divider()
                with st.expander("Prompt sent for generation"):
                    # The prompt is the context-construction step: question plus chunks.
                    st.code(answered["prompt"])

st.divider()
st.subheader("Evaluation")
st.caption(
    "Runs 24 labeled questions at chunk presets A, B, and C. "
    "Precision, recall, and MRR ignore unanswerable questions. "
    "Those are scored as abstention. With a Gemini key this calls the model "
    "once per question per preset."
)
if st.button("Run evaluation"):
    try:
        with st.spinner("Scoring the question set. This can take a few minutes."):
            report = requests.post(f"{API_URL}/evaluate", timeout=600)
            report.raise_for_status()
            report = report.json()
    except requests.RequestException as exc:
        st.error(f"Evaluation failed: {exc}")
    else:
        summary = [
            {
                "Preset": item["name"],
                "Chunk size": item["chunk_size"],
                "Overlap": item["chunk_overlap"],
                "Precision@3": item["precision_at_3"],
                "Precision@5": item["precision_at_5"],
                "Recall@5": item["recall_at_5"],
                "MRR": item["mrr"],
                "Answer accuracy": item["answer_accuracy"],
                "Abstention accuracy": item["abstention_accuracy"],
            }
            for item in report["presets"]
        ]
        st.dataframe(summary, use_container_width=True)
        st.markdown("**Failure analysis**")
        if not report["failures"]:
            st.write("No failed questions in this run.")
        for failure in report["failures"]:
            with st.expander(f"{failure['preset']} · {failure['id']} · {failure['failure']}"):
                st.write(failure["question"])
                st.write(f"Stage: {failure['stage']}")
                st.write(failure["what"])
                st.write(failure["why"])
                st.write(f"Possible fix: {failure['fix']}")
        for preset in report["presets"]:
            with st.expander(f"Questions · preset {preset['name']}"):
                table = [
                    {
                        "Id": row["id"],
                        "Type": row["type"],
                        "Question": row["question"],
                        "Retrieved": ", ".join(item["title"] for item in row["retrieved"]),
                        "Answer": row["answer"][:180],
                        "Expected": row["expected_answer"],
                        "Correct": row["correct"],
                        "Grounded": row["grounded"],
                        "Citation": row["citation_correct"],
                        "Failure": row["failure"],
                    }
                    for row in preset["rows"]
                ]
                st.dataframe(table, use_container_width=True)
