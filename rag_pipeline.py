"""The complete, deliberately linear Retrieval-Augmented Generation workflow."""

import re
import shutil
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

import config
from document_processor import chunk_documents, load_documents
from llm_client import (
    create_chat_model,
    create_embeddings,
    get_default_model,
    get_embedding_model_name,
)


class RAGPipeline:
    """Connect document processing, FAISS retrieval, and answer generation.

    This single class holds the models and vector store used during a Streamlit
    session. Its methods follow the RAG diagram in the README from top to bottom.
    """

    def __init__(
        self,
        provider: str = config.LLM_PROVIDER,
        model_name: str | None = None,
        temperature: float = config.TEMPERATURE,
        top_k: int = config.TOP_K,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> None:
        """Create a pipeline using simple configuration values.

        Args:
            provider: ``openai`` or ``gemini``.
            model_name: Chat model name, or ``None`` to use the configured default.
            temperature: Creativity setting for query rewriting and answers.
            top_k: Chunks retrieved before relevance filtering.
            similarity_threshold: Minimum normalized relevance score from 0 to 1.

        Returns:
            Nothing. The initialized object is ready to load or build FAISS.
        """
        config.validate_settings(
            provider,
            temperature,
            top_k,
            similarity_threshold,
        )
        self.provider = provider
        self.model_name = model_name or get_default_model(provider)
        self.temperature = temperature
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embeddings = create_embeddings(provider)
        self.chat_model = create_chat_model(
            provider,
            self.model_name,
            temperature,
        )
        self.vector_store: FAISS | None = None

    def _store_path(self) -> Path:
        """Return a separate FAISS folder for the selected embedding model.

        Different embedding models produce incompatible vector spaces. Keeping
        each model in its own folder prevents an OpenAI query vector from being
        compared with a Gemini document index.

        Returns:
            The local folder used to save this pipeline's FAISS files.
        """
        model_name = get_embedding_model_name(self.provider)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_name)
        # Replace any character in the model name that is not a letter, digit, underscore, or hyphen with an underscore 
        # to create a safe folder/file name.

        return config.VECTOR_STORE_DIR / f"{self.provider}_{safe_name}"
    #VECTOR_STORE_DIR = BASE_DIR / "vector_store"

    def build_vector_store(self) -> int:
        """Load source files, chunk them, embed them, and save a FAISS index.

        FAISS is a fast local similarity-search library. It stores vectors on
        this computer, so this teaching project needs no vector database server.

        Returns:
            The number of chunks added to the new index.
        """
        documents = load_documents(config.DATA_DIR) 
        #config.DATA_DIR is the path where documents exists inside data folder
        
        chunks = chunk_documents(
            documents,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP,
        )
        
        if not chunks:
            raise ValueError("No chunks were created from the source documents.")
        # Normalizing vectors lets us convert FAISS squared L2 distance into
        # cosine similarity with a small, explicit formula during retrieval.
        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings,
            normalize_L2=True,
        )
        store_path = self._store_path()
        store_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(store_path))
        return len(chunks)

    def load_or_build_vector_store(self) -> str:
        """Load an existing local index or automatically build the first one.

        Returns:
            A short status string suitable for logs or the Streamlit interface.
        """
        store_path = self._store_path()
        index_file = store_path / "index.faiss"
        
        '''🔹 index.faiss is the binary file where FAISS actual vector index store is done.
        👉 Without this file, FAISS search will not work as vectors will be missing'''
        
        metadata_file = store_path / "index.pkl" #for metadata
        '''🔹 index.pkl is used to store metadata.
👉 Without this file, FAISS will only return vectors but will not be able to return metadata information'''
        
        if not index_file.exists() or not metadata_file.exists():
            chunk_count = self.build_vector_store() #an object method can be called by an object so self. is used
            return f"Built a new vector store with {chunk_count} chunks."

        # The pickle contains only metadata this app previously wrote locally.
        # Never enable this option for vector stores downloaded from strangers.
        self.vector_store = FAISS.load_local(
            str(store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
        )
        return "Loaded the existing vector store."

    def rebuild_vector_store(self) -> int:
        """Delete this embedding model's old index and build it again.

        Returns:
            The number of chunks in the rebuilt index as build_vector_store is returning len(chunks) as created above.
        """
        store_path = self._store_path()
        if store_path.exists():
            shutil.rmtree(store_path)
        return self.build_vector_store()

    def rewrite_query(
        self,
        question: str,
        chat_history: list[dict[str, Any]],
    ) -> str:
        """Turn a conversational follow-up into a standalone search query.

        Query rewriting is an Advanced RAG step. For example, it can replace
        "its" in a follow-up with the subject mentioned in an earlier message.
        A first question needs no rewriting and is returned unchanged.

        Args:
            question: User's newest question.
            chat_history: Earlier user and assistant messages.

        Returns:
            A concise standalone query used only for retrieval.
        """
        if not chat_history:
            return question

        recent_messages = chat_history[-6:] #last 6 messages
        history_text = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in recent_messages
        )
        prompt = (
            "Rewrite the latest question as one standalone search query. "
            "Resolve pronouns using the conversation. Preserve the user's meaning. "
            "Return only the rewritten query.\n\n"
            f"Conversation:\n{history_text}\n\nLatest question: {question}"
        )
        response = self.chat_model.invoke([HumanMessage(content=prompt)])
        rewritten = str(response.content).strip()
        return rewritten or question

    def retrieve_documents(self, query: str) -> list[tuple[Document, float]]:
        """Search FAISS and return chunks with cosine similarity scores.

        FAISS returns squared L2 distance, where smaller is better. Because this
        project normalizes every vector, cosine similarity is exactly
        ``1 - (squared_distance / 2)``. The result normally ranges from -1 to 1,
        and larger values mean more similar meaning.

        Args:
            query: Standalone retrieval query.

        Returns:
            Up to ``top_k`` pairs of document chunk and cosine similarity.
        """
        if self.vector_store is None:
            raise RuntimeError("Load or build the vector store before searching.")
        distance_results = self.vector_store.similarity_search_with_score(
            query,
            k=self.top_k,
        )
        return [
            (document, 1.0 - (float(squared_distance) / 2.0))
            for document, squared_distance in distance_results
        ]

    def filter_documents(
        self,
        results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        """Keep only chunks that meet the configured relevance threshold.

        Cosine similarity has clear mathematics, but useful score ranges still
        vary by embedding model and content. The threshold therefore needs
        tuning rather than being treated as a universal value.

        Args:
            results: Retrieved chunks paired with relevance scores.

        Returns:
            Only result pairs whose scores meet or exceed the threshold.
        """
        return [
            (document, score)
            for document, score in results
            if score >= self.similarity_threshold
        ]

    def build_context(self, results: list[tuple[Document, float]]) -> str:
        """Format retrieved chunks as labeled context for the chat model.

        Args:
            results: Filtered chunks paired with relevance scores.

        Returns:
            One text block containing source labels and chunk content.
        """
        context_parts = []
        for document, _score in results: #results is list[tuple of document and scores (float) 
            metadata = document.metadata
            label = metadata["source"]
            if metadata.get("page"):
                label += f", page {metadata['page']}"
            if metadata.get("section"):
                label += f", section {metadata['section']}"
            context_parts.append(
                f"[Source: {label}]\n{document.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, question: str, context: str) -> str:
        """Ask the chat model to answer only from retrieved context.

        Args:
            question: User's original wording, used for the natural final answer.
            context: Relevant source chunks with citation labels.

        Returns:
            A grounded answer with inline source labels where possible.
        """
        system_prompt = (
            "You answer questions about the fictional company NovaTech Solutions. "
            "Use only the provided context. Do not use outside knowledge or invent facts. "
            "Cite supporting source labels in your answer. If the answer is absent, say: "
            "'I could not find that information in the NovaTech knowledge base.'"
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content).strip()

    def format_sources(
        self,
        results: list[tuple[Document, float]],
    ) -> list[dict[str, Any]]:
        """Create deduplicated source records for display in Streamlit.

        Args:
            results: Filtered chunks actually supplied to the answer model.

        Returns:
            Source dictionaries containing filename, location, and relevance.
        """
        sources: list[dict[str, Any]] = []
        seen_locations: set[tuple[Any, ...]] = set()
        for document, score in results:
            metadata = document.metadata
            location = (
                metadata["source"],
                metadata.get("page"),
                metadata.get("section"),
            )
            if location in seen_locations:
                continue
            seen_locations.add(location)
            sources.append(
                {
                    "source": metadata["source"],
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "score": round(float(score), 3),
                }
            )
        return sources

    def ask(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the visible RAG sequence and return an answer plus evidence.

        Args:
            question: User's newest question.
            chat_history: Earlier messages used only to rewrite follow-ups.

        Returns:
            A dictionary with the answer, rewritten query, and actual sources.
        """
        if not question.strip():
            raise ValueError("Please enter a non-empty question.")
        history = chat_history or []
        rewritten_query = self.rewrite_query(question.strip(), history)
        retrieved = self.retrieve_documents(rewritten_query)
        filtered = self.filter_documents(retrieved)

        if not filtered:
            return {
                "answer": (
                    "I could not find that information in the NovaTech knowledge base."
                ),
                "rewritten_query": rewritten_query,
                "sources": [],
            }

        context = self.build_context(filtered)
        answer = self.generate_answer(question.strip(), context)
        return {
            "answer": answer,
            "rewritten_query": rewritten_query,
            "sources": self.format_sources(filtered),
        }
