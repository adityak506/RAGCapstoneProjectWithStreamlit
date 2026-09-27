"""Streamlit chat interface for the beginner-friendly NovaTech RAG project."""

from typing import Any

import streamlit as st

import config
from llm_client import get_default_model
from rag_pipeline import RAGPipeline


st.set_page_config(page_title="NovaTech RAG Assistant", page_icon="💬")
st.title("NovaTech Knowledge Assistant")
st.caption("Ask grounded questions across the company handbook, AI guide, and policies.")


def display_sources(sources: list[dict[str, Any]]) -> None:
    """Show only the source locations used to generate an answer.

    Args:
        sources: Source dictionaries returned by the RAG pipeline.

    Returns:
        Nothing. The function writes source labels to the Streamlit page.
    """
    if not sources:
        return
    with st.expander("Sources", expanded=True):
        for source in sources:
            location = source["source"]
            if source.get("page"):
                location += f", page {source['page']}"
            if source.get("section"):
                location += f", section {source['section']}"
            st.markdown(f"- {location} · relevance `{source['score']:.3f}`")


@st.cache_resource(show_spinner=False)
def get_pipeline(
    provider: str,
    model_name: str,
    temperature: float,
    top_k: int,
    similarity_threshold: float,
) -> RAGPipeline:
    """Create and cache models so Streamlit reruns do not reconnect every time.

    Args:
        provider: ``openai`` or ``gemini``.
        model_name: Chat model selected in the sidebar.
        temperature: Model creativity control.
        top_k: Number of chunks retrieved before filtering.
        similarity_threshold: Minimum cosine similarity accepted as context.

    Returns:
        A ready-to-use RAG pipeline with its FAISS index loaded.
    """
    pipeline = RAGPipeline(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    pipeline.load_or_build_vector_store()
    return pipeline


with st.sidebar:
    st.header("RAG settings")
    provider = st.selectbox(
        "LLM provider",
        ["openai", "gemini"],
        index=0 if config.LLM_PROVIDER == "openai" else 1,
    )
    model_name = st.text_input(
        "Chat model",
        value=get_default_model(provider),
        key=f"model_{provider}",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(config.TEMPERATURE),
        step=0.1,
    )
    top_k = st.slider("Top-K retrieval", 1, 10, config.TOP_K)
    similarity_threshold = st.slider(
        "Cosine similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(config.SIMILARITY_THRESHOLD),
        step=0.05,
        help="Tune this per embedding model. Higher values pass fewer chunks.",
    )

try:
    pipeline = get_pipeline(
        provider,
        model_name,
        temperature,
        top_k,
        similarity_threshold,
    )
except Exception as error:
    st.error(f"Could not initialize the RAG pipeline: {error}")
    st.info("Check your .env API key and model names, then refresh the app.")
    st.stop()

with st.sidebar:
    if st.button("Rebuild Vector Store", use_container_width=True):
        try:
            with st.spinner("Reprocessing documents and creating embeddings..."):
                chunk_count = pipeline.rebuild_vector_store()
            st.success(f"Rebuilt the index with {chunk_count} chunks.")
        except Exception as error:
            st.error(f"Rebuild failed: {error}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("rewritten_query"):
                st.caption(f"Retrieval query: {message['rewritten_query']}")
            display_sources(message.get("sources", []))

question = st.chat_input("Ask about NovaTech Solutions...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the knowledge base..."):
                result = pipeline.ask(
                    question,
                    chat_history=st.session_state.messages[:-1],
                )
            st.markdown(result["answer"])
            if result["rewritten_query"] != question:
                st.caption(f"Retrieval query: {result['rewritten_query']}")
            display_sources(result["sources"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "rewritten_query": (
                        result["rewritten_query"]
                        if result["rewritten_query"] != question
                        else ""
                    ),
                    "sources": result["sources"],
                }
            )
        except Exception as error:
            st.error(f"I could not complete the request: {error}")