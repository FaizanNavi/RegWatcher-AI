import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional
from ..utils.config import POSTGRES_URL

logger = logging.getLogger(__name__)

def init_database():
    engine = create_engine(POSTGRES_URL)
    logger.info("Initializing PostgreSQL database with full schema")
    
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                document_number TEXT PRIMARY KEY,
                abstract TEXT,
                action TEXT,
                agency_names TEXT,
                amendatory_instructions TEXT,
                body_html_url TEXT,
                cfr_references TEXT,
                cfr_topics TEXT,
                citation TEXT,
                comment_url TEXT,
                comments_close_on TEXT,
                correction_of TEXT,
                corrections TEXT,
                dates TEXT,
                disposition_notes TEXT,
                docket_id TEXT,
                docket_ids TEXT,
                dockets TEXT,
                effective_on TEXT,
                end_page TEXT,
                excerpts TEXT,
                executive_order_notes TEXT,
                executive_order_number TEXT,
                explanation TEXT,
                full_text_xml_url TEXT,
                html_url TEXT,
                images TEXT,
                images_metadata TEXT,
                json_url TEXT,
                mods_url TEXT,
                not_received_for_publication TEXT,
                page_length TEXT,
                page_views TEXT,
                pdf_url TEXT,
                president TEXT,
                presidential_document_number TEXT,
                proclamation_number TEXT,
                public_inspection_pdf_url TEXT,
                publication_date TEXT,
                raw_text_url TEXT,
                regulation_id_number_info TEXT,
                regulation_id_numbers TEXT,
                regulations_dot_gov_info TEXT,
                regulations_dot_gov_url TEXT,
                related_documents TEXT,
                significant TEXT,
                signing_date TEXT,
                start_page TEXT,
                subtype TEXT,
                title TEXT,
                toc_doc TEXT,
                toc_subject TEXT,
                topics TEXT,
                type TEXT,
                volume TEXT
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agencies (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_agencies (
                document_number TEXT,
                agency_id INTEGER,
                PRIMARY KEY (document_number, agency_id),
                FOREIGN KEY (document_number) REFERENCES documents(document_number),
                FOREIGN KEY (agency_id) REFERENCES agencies(id)
            )
        """))
    logger.info("PostgreSQL Database schema initialized.")

class PostgresManager:
    def __init__(self):
        try:
            init_database()
            self.engine = create_engine(POSTGRES_URL)
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.engine = None

    def get_schema(self) -> str:
        if not self.engine:
            return "Database not connected."
        logger.info("Extracting schema for SQL agent")
        schema_text = ""
        try:
            with self.engine.connect() as conn:
                tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
                for (table_name,) in tables:
                    columns = conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")).fetchall()
                    col_desc = ", ".join([f"{c[0]} ({c[1]})" for c in columns])
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()[0]
                    schema_text += f"Table: {table_name} ({count} rows)\n  Columns: {col_desc}\n\n"
            return schema_text
        except Exception as e:
            logger.error(f"Error extracting schema: {e}")
            return "Error extracting schema."

    def execute_query(self, sql: str) -> Optional[pd.DataFrame]:
        if not self.engine:
            return None
        logger.info(f"Executing SQL query: {sql[:100]}...")
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(text(sql), conn)
            logger.info(f"Query returned {len(df)} rows")
            return df.head(500)
        except Exception as e:
            logger.error(f"SQL Execution Error: {e}")
            raise e
