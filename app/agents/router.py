import json
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..utils.config import GROQ_API_KEY, LLM_MODEL, GROQ_BASE_URL
logger = logging.getLogger(__name__)
class RouterAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GROQ_API_KEY,
            openai_api_base=GROQ_BASE_URL,
            temperature=0,
        )
    def route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        logger.info(f"Router: classifying query: {query[:80]}...")
        prompt = ChatPromptTemplate.from_template(
            "Classify this query about US federal regulations into one of these types:\n\n"
            "1. 'sql' — needs exact document lookup (mentions doc numbers, specific dates, exact agency names)\n"
            "2. 'rag' — needs semantic understanding (asks about topics, concepts, summaries)\n"
            "3. 'hybrid' — needs both exact data AND semantic understanding\n\n"
            "Query: {query}\n\n"
            "Return ONLY one word: sql, rag, or hybrid"
        )
        response = (prompt | self.llm).invoke({"query": query})
        route = response.content.strip().lower()
        if route not in ("sql", "rag", "hybrid"):
            route = "rag"
        logger.info(f"Router: classified as '{route}'")
        return {"route": route, "status": f"Routed to: {route}"}
