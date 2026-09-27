import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def _read_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number.") from error
    
def _read_float(name: str, default: int) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number.") from error    
    
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

TEMPERATURE = _read_float("TEMPERATURE", 0.2)
TOP_K = _read_int("TOP_K", 3)
SIMILARITY_THRESHOLD = _read_float("SIMILARITY_THRESHOLD", 0.35)
CHUNK_SIZE = _read_int("CHUNK_SIZE", 900)
CHUNK_OVERLAP = _read_int("CHUNK_OVERLAP", 150)


def validate_settings(
        provider: str,
        temperature: float,
        top_k: int,
        similarity_threshold = float,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
) -> None:
    if provider not in {"openai", "gemini"}:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'gemini'")
    if not 0<= temperature <=2:
        raise ValueError("TEMPERATURE must be between 0 and 2")
    if top_k<1:
        raise ValueError("top_k must be at least 1")
    if not 0<= similarity_threshold <=2:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if chunk_size < 100:
        raise ValueError("CHUNK_SIZE must be at least 100 chars.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than chunk_size")