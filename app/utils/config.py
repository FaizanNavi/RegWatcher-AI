import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "regulations")
FEDERAL_API_BASE = "https://www.federalregister.gov/api/v1"

POSTGRES_USER = os.getenv("POSTGRES_USER", "regwatcher")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "regpassword")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "regwatcher_db")
POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MAX_CRITIC_CYCLES = int(os.getenv("MAX_CRITIC_CYCLES", "2"))
MIN_CRITIC_SCORE = int(os.getenv("MIN_CRITIC_SCORE", "7"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
