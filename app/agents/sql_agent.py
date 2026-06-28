import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..database.postgres_manager import PostgresManager
from ..utils.config import GROQ_API_KEY, LLM_MODEL, GROQ_BASE_URL

logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GROQ_API_KEY,
            openai_api_base=GROQ_BASE_URL,
            temperature=0,
        )
        self.db = PostgresManager()

    def query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        logger.info(f"SQL Agent: generating query for '{query[:60]}...'")
        
        schema = self.db.get_schema()
        
        prompt = ChatPromptTemplate.from_template(
            "You are an expert SQL analyst. Generate a SELECT query for the question.\n\n"
            "Database Schema:\n{schema}\n\n"
            "Question: {query}\n\n"
            "Rules:\n"
            "- Only generate a SELECT statement.\n"
            "- Ensure the query is valid PostgreSQL.\n"
            "- Return ONLY the raw SQL, no markdown formatting."
        )
        
        response = (prompt | self.llm).invoke({"schema": schema, "query": query})
        sql_query = response.content.strip().strip("`").strip()
        if sql_query.startswith("sql"):
            sql_query = sql_query[3:].strip()
            
        logger.info(f"Generated SQL: {sql_query}")
        
        try:
            df = self.db.execute_query(sql_query)
            if df is None or df.empty:
                result_str = "No results found from the database."
            else:
                result_str = df.to_string(index=False, max_rows=20)
                logger.info(f"Extracted {len(df)} rows from SQL execution")
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            result_str = f"Database query failed: {str(e)}"
            
        return {
            "retrieved_chunks": [{"text": f"SQL Query Results:\n{result_str}", "metadata": {"source": "sql"}}],
            "status": "SQL context generated"
        }
