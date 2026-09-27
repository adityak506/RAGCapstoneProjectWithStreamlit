import os
from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import config

def require_api_key(provider: str) -> str:
    variable_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.getenv(variable_name, "").strip()
    if not api_key:
        raise ValueError(f"{variable_name} is missing. Please review your .env file")
    return api_key

def get_default_model(provider: str) -> str:
    return config.OPENAI_MODEL if provider == "openai" else config.GEMINI_MODEL

def get_embedding_model_name(provider: str) -> str:
    if provider == "openai":
        return config.OPENAI_EMBEDDING_MODEL
    return config.GEMINI_EMBEDDING_MODEL

def create_chat_model(
        provider: str,
        model_name: str,
        temperature: float,
) -> Any:
    api_key = require_api_key(provider)
    if provider == "openai":
        return ChatOpenAI(
            model = model_name,
            temperature = temperature,
            api_key = api_key
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model = model_name,
            temperature = temperature,
            google_api_key = api_key,
        )
    raise ValueError("Provider must be 'openai' or 'gemini'")

def create_embeddings(provider: str) -> Any:
    api_key = require_api_key(provider)
    if provider == "openai":
        return OpenAIEmbeddings(
            model = config.OPENAI_EMBEDDING_MODEL,
            api_key = api_key,
        )
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model = config.GEMINI_EMBEDDING_MODEL,
            google_api_key = api_key,
        )
    raise ValueError("Provider must be 'openai' or 'gemini'.")