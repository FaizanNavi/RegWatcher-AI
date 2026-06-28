"""
Federal Register Data Pipeline
===============================
Complete ETL pipeline that:
  1. Fetches documents from the US Federal Register API
  2. Saves raw JSON organized by publication date
  3. Converts to CSV organized by publication date
  4. Runs OCR on attached PDFs for full-text extraction
  5. Chunks text and stores in ChromaDB for semantic search
  6. Loads structured data into PostgreSQL

Folder Structure:
  data/
  ├── raw/json/YYYY-MM-DD/          (one .json per document)
  ├── processed/csv/YYYY-MM-DD/     (documents.csv, agencies.csv, document_agencies.csv)
  ├── consolidated/                  (merged CSVs across all dates for DB load)
  └── chromadb/                      (vector store persistence)
"""

import json
import hashlib
import requests
import pandas as pd
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import logging
import time
import argparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEDERAL_API_BASE = "https://www.federalregister.gov/api/v1"
DATA_DIR = Path("./data")
RAW_JSON_DIR = DATA_DIR / "raw" / "json"
PROCESSED_CSV_DIR = DATA_DIR / "processed" / "csv"
CONSOLIDATED_DIR = DATA_DIR / "consolidated"

# ---------------------------------------------------------------------------
# All 55+ metadata fields exposed by the Federal Register API
# ---------------------------------------------------------------------------
ALL_FIELDS = [
    "abstract", "action", "agencies", "agency_names", "amendatory_instructions",
    "body_html_url", "cfr_references", "cfr_topics", "citation", "comment_url",
    "comments_close_on", "correction_of", "corrections", "dates", "disposition_notes",
    "docket_id", "docket_ids", "dockets", "document_number", "effective_on", "end_page",
    "excerpts", "executive_order_notes", "executive_order_number", "explanation",
    "full_text_xml_url", "html_url", "images", "images_metadata", "json_url", "mods_url",
    "not_received_for_publication", "page_length", "page_views", "pdf_url", "president",
    "presidential_document_number", "proclamation_number", "public_inspection_pdf_url",
    "publication_date", "raw_text_url", "regulation_id_number_info", "regulation_id_numbers",
    "regulations_dot_gov_info", "regulations_dot_gov_url", "related_documents", "significant",
    "signing_date", "start_page", "subtype", "title", "toc_doc", "toc_subject", "topics",
    "type", "volume",
]


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def fetch_documents_by_date(
    date: str, page: int = 1, per_page: int = 1000, agency: Optional[str] = None, president: Optional[str] = None
) -> tuple[List[Dict], int]:
    """Fetch a page of documents for a specific date from the API, returning (results, total_pages)."""
    params = {
        "per_page": per_page,
        "page": page,
        "conditions[publication_date][is]": date,
        "fields[]": ALL_FIELDS,
    }
    if agency:
        params["conditions[agencies][]"] = agency
    if president:
        params["conditions[president][]"] = president

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {date}, page {page} (Attempt {attempt + 1}/{max_retries})")
            resp = requests.get(f"{FEDERAL_API_BASE}/documents", params=params, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Got status {resp.status_code} for {date}, page {page}")
                return [], 1
                
            data = resp.json()
            results = data.get("results", [])
            total_pages = data.get("total_pages", 1)
            logger.info(f"Successfully fetched {len(results)} documents from page {page}/{total_pages}")
            return results, total_pages
        except Exception as e:
            logger.warning(f"Error fetching page {page} for {date}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to fetch {date}, page {page} after {max_retries} attempts")
                return [], 1
    return [], 1


def extract_text_from_pdf(pdf_url: str) -> str:
    """Download a PDF and extract text using PyMuPDF + Tesseract OCR fallback."""
    if not pdf_url:
        return ""
    try:
        logger.info(f"Downloading PDF for OCR: {pdf_url}")
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        doc = fitz.open(stream=resp.content, filetype="pdf")
        text = ""
        for page in doc:
            page_text = page.get_text()
            if len(page_text.strip()) > 50:
                text += page_text
            else:
                # Scanned page — fall back to Tesseract OCR
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text += pytesseract.image_to_string(img)
        logger.info(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"OCR Pipeline Error for {pdf_url}: {e}")
        return ""


def chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """Split text into overlapping chunks with intelligent boundary detection."""
    if not text or not text.strip():
        return []
    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at a natural boundary
            for sep in ["\n\n", "\n", ". ", " "]:
                brk = text.rfind(sep, start + chunk_size // 2, end)
                if brk != -1:
                    end = brk + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def safe_json(val) -> str:
    """Safely serialize a value to a JSON-friendly string."""
    if val is None:
        return ""
    if isinstance(val, (dict, list)):
        return json.dumps(val)
    return str(val)


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline Class
# ═══════════════════════════════════════════════════════════════════════════

class FederalRegisterPipeline:
    """
    End-to-end data pipeline for the Federal Register.

    Usage:
        pipeline = FederalRegisterPipeline(retrieval_agent=retrieval)
        stats = pipeline.ingest(pages=5, per_page=50)
    """

    def __init__(
        self,
        retrieval_agent=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.retrieval = retrieval_agent
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure base directories exist
        for d in [RAW_JSON_DIR, PROCESSED_CSV_DIR, CONSOLIDATED_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info("Federal Register Data Pipeline initialized")

    # ───────────────────────────────────────────────────────────────────
    # Step 1: Fetch
    # ───────────────────────────────────────────────────────────────────
    def _fetch_all(
        self, start_date: Optional[str], end_date: Optional[str], agency: Optional[str], president: Optional[str]
    ) -> List[Dict]:
        """Fetch all pages for a given date range."""
        dates = []
        from datetime import datetime, timedelta
        
        if start_date is None and end_date is None:
            end = datetime.now()
            start = end - timedelta(days=3)
            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')
            
        if end_date is None:
            dates = [start_date]
        else:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            current = start
            while current <= end:
                dates.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)

        logger.info(f"Going to download {len(dates)} dates")

        all_docs: List[Dict] = []
        for date in dates:
            page = 1
            total_pages = 1
            while page <= total_pages:
                docs, updated_total_pages = fetch_documents_by_date(
                    date=date, page=page, agency=agency, president=president
                )
                total_pages = max(total_pages, updated_total_pages)
                
                if not docs:
                    logger.warning(f"No more documents found for {date}, page {page}")
                    break
                    
                all_docs.extend(docs)
                page += 1
                time.sleep(1) # Be nice to the API
                
            if len(dates) > 1:
                time.sleep(1)
                
        return all_docs

    # ───────────────────────────────────────────────────────────────────
    # Step 2: Save raw JSON — date-wise folders
    # ───────────────────────────────────────────────────────────────────
    def _save_raw_json(
        self, docs_by_date: Dict[str, List[Dict]]
    ) -> int:
        """
        Save each raw API document as an individual JSON file:
            data/raw/json/YYYY-MM-DD/<document_number>.json
        """
        total_saved = 0
        for pub_date, docs in docs_by_date.items():
            date_dir = RAW_JSON_DIR / pub_date
            date_dir.mkdir(parents=True, exist_ok=True)
            for doc in docs:
                doc_number = doc.get("document_number", "unknown")
                # Sanitize filename
                safe_name = doc_number.replace("/", "_").replace("\\", "_")
                filepath = date_dir / f"{safe_name}.json"
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
                total_saved += 1
            logger.info(
                f"Saved {len(docs)} raw JSON files to {date_dir}"
            )
        return total_saved

    # ───────────────────────────────────────────────────────────────────
    # Step 3: Convert to CSV — date-wise folders
    # ───────────────────────────────────────────────────────────────────
    def _save_csv_by_date(
        self, docs_by_date: Dict[str, List[Dict]]
    ) -> int:
        """
        For each date, produce three CSV files:
            data/processed/csv/YYYY-MM-DD/documents.csv
            data/processed/csv/YYYY-MM-DD/agencies.csv
            data/processed/csv/YYYY-MM-DD/document_agencies.csv
        """
        total_rows = 0
        for pub_date, docs in docs_by_date.items():
            date_dir = PROCESSED_CSV_DIR / pub_date
            date_dir.mkdir(parents=True, exist_ok=True)

            doc_records, agencies_dict, doc_agencies = self._extract_tables(docs)

            pd.DataFrame(doc_records).to_csv(
                date_dir / "documents.csv", index=False
            )
            pd.DataFrame(
                [{"id": k, "name": v} for k, v in agencies_dict.items()]
            ).to_csv(date_dir / "agencies.csv", index=False)
            pd.DataFrame(doc_agencies).to_csv(
                date_dir / "document_agencies.csv", index=False
            )

            total_rows += len(doc_records)
            logger.info(
                f"Saved CSV for {pub_date}: "
                f"{len(doc_records)} docs, "
                f"{len(agencies_dict)} agencies"
            )
        return total_rows

    # ───────────────────────────────────────────────────────────────────
    # Step 4: Build consolidated CSVs (merged across all dates)
    # ───────────────────────────────────────────────────────────────────
    def _save_consolidated_csv(
        self, all_docs: List[Dict]
    ) -> None:
        """
        Merge every document into a single set of CSVs for PostgreSQL loading:
            data/consolidated/all_documents.csv
            data/consolidated/all_agencies.csv
            data/consolidated/all_document_agencies.csv
        """
        doc_records, agencies_dict, doc_agencies = self._extract_tables(all_docs)

        pd.DataFrame(doc_records).to_csv(
            CONSOLIDATED_DIR / "all_documents.csv", index=False
        )
        pd.DataFrame(
            [{"id": k, "name": v} for k, v in agencies_dict.items()]
        ).to_csv(CONSOLIDATED_DIR / "all_agencies.csv", index=False)
        pd.DataFrame(doc_agencies).to_csv(
            CONSOLIDATED_DIR / "all_document_agencies.csv", index=False
        )
        logger.info(
            f"Consolidated CSVs saved: "
            f"{len(doc_records)} documents, "
            f"{len(agencies_dict)} agencies, "
            f"{len(doc_agencies)} doc-agency links"
        )

    # ───────────────────────────────────────────────────────────────────
    # Step 5: OCR + Chunking → ChromaDB
    # ───────────────────────────────────────────────────────────────────
    def _vectorize_documents(self, all_docs: List[Dict]) -> int:
        """
        For each document:
          - Combine title + abstract + PDF OCR text
          - Chunk the text
          - Embed and store in ChromaDB via the RetrievalAgent
        Returns total chunks created.
        """
        if not self.retrieval:
            logger.warning(
                "No retrieval agent provided — skipping ChromaDB vectorization"
            )
            return 0

        total_chunks = 0
        for i, doc in enumerate(all_docs, 1):
            doc_number = doc.get("document_number", "unknown")
            title = doc.get("title", "")
            abstract = doc.get("abstract", "") or ""
            pdf_url = doc.get("pdf_url", "")

            # Build combined text
            text_content = f"Title: {title}\n\n{abstract}"
            if pdf_url:
                pdf_text = extract_text_from_pdf(pdf_url)
                if pdf_text:
                    text_content += f"\n\nFull Text:\n{pdf_text}"

            if len(text_content.strip()) <= 20:
                continue

            chunks = chunk_text(
                text_content, self.chunk_size, self.chunk_overlap
            )
            if not chunks:
                continue

            metadatas = [
                {"source": doc_number, "chunk_index": ci}
                for ci in range(len(chunks))
            ]
            ids = [
                hashlib.md5(f"{doc_number}:{ci}".encode()).hexdigest()[:16]
                for ci in range(len(chunks))
            ]
            self.retrieval.add_documents(chunks, metadatas, ids)
            total_chunks += len(chunks)

            if i % 10 == 0:
                logger.info(
                    f"Vectorized {i}/{len(all_docs)} documents "
                    f"({total_chunks} chunks so far)"
                )

        logger.info(f"Vectorization complete: {total_chunks} total chunks")
        return total_chunks

    # ───────────────────────────────────────────────────────────────────
    # Step 6: Load to PostgreSQL
    # ───────────────────────────────────────────────────────────────────
    def _load_to_postgres(self) -> bool:
        """Load the consolidated CSVs into PostgreSQL tables."""
        logger.info("Loading consolidated CSV data into PostgreSQL database")
        try:
            from sqlalchemy import create_engine
            from ..utils.config import POSTGRES_URL

            engine = create_engine(POSTGRES_URL)

            docs_csv = CONSOLIDATED_DIR / "all_documents.csv"
            agencies_csv = CONSOLIDATED_DIR / "all_agencies.csv"
            doc_agencies_csv = CONSOLIDATED_DIR / "all_document_agencies.csv"

            with engine.begin() as conn:
                if docs_csv.exists():
                    pd.read_csv(docs_csv).to_sql(
                        "documents", conn, if_exists="replace", index=False
                    )
                if agencies_csv.exists():
                    pd.read_csv(agencies_csv).to_sql(
                        "agencies", conn, if_exists="replace", index=False
                    )
                if doc_agencies_csv.exists():
                    pd.read_csv(doc_agencies_csv).to_sql(
                        "document_agencies", conn, if_exists="replace", index=False
                    )

            logger.info("PostgreSQL database load complete")
            return True
        except Exception as e:
            logger.error(f"Failed to load into PostgreSQL: {e}")
            return False

    # ───────────────────────────────────────────────────────────────────
    # Helper: extract relational tables from raw documents
    # ───────────────────────────────────────────────────────────────────
    def _extract_tables(
        self, docs: List[Dict]
    ) -> tuple:
        """
        Parse raw API documents into three relational tables:
          - doc_records   (flat row per document, all metadata fields)
          - agencies_dict (agency_id → agency_name)
          - doc_agencies  (document_number, agency_id)
        """
        doc_records: List[Dict] = []
        agencies_dict: Dict[int, str] = {}
        doc_agencies: List[Dict] = []

        for doc in docs:
            doc_number = doc.get("document_number", "unknown")

            # Flatten every field except 'agencies'
            record = {}
            for field in ALL_FIELDS:
                if field != "agencies":
                    record[field] = safe_json(doc.get(field))
            doc_records.append(record)

            # Extract agency relationships
            for a in doc.get("agencies", []):
                if isinstance(a, dict):
                    a_id = a.get("id")
                    a_name = a.get("name")
                    if a_id and a_name:
                        agencies_dict[a_id] = a_name
                        doc_agencies.append(
                            {"document_number": doc_number, "agency_id": a_id}
                        )

        return doc_records, agencies_dict, doc_agencies

    # ───────────────────────────────────────────────────────────────────
    # Public API: Download & Preprocess
    # ───────────────────────────────────────────────────────────────────
    def download(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        agency: Optional[str] = None,
        president: Optional[str] = None,
    ) -> Dict:
        """
        Phase 1: Fetch documents from Federal Register API and save raw JSON.
        """
        start_time = time.time()
        logger.info(
            f"{'='*60}\n"
            f"  PIPELINE START — DOWNLOAD ({start_date or 'default'} to {end_date or 'default'})\n"
            f"{'='*60}"
        )

        # ── 1. Fetch ─────────────────────────────────────────────────
        logger.info("[Step 1/2] Fetching documents from Federal Register API...")
        all_docs = self._fetch_all(start_date, end_date, agency, president)
        if not all_docs:
            logger.error("No documents fetched. Download aborted.")
            return {"documents_fetched": 0, "error": "No documents returned"}

        # ── 2. Group by publication_date & Save ──────────────────────
        logger.info("[Step 2/2] Grouping by date and saving raw JSON...")
        docs_by_date: Dict[str, List[Dict]] = defaultdict(list)
        for doc in all_docs:
            pub_date = doc.get("publication_date", "unknown")
            docs_by_date[pub_date].append(doc)

        date_count = len(docs_by_date)
        json_saved = self._save_raw_json(docs_by_date)
        
        elapsed = round(time.time() - start_time, 2)
        
        logger.info(
            f"\n{'='*60}\n"
            f"  DOWNLOAD COMPLETE in {elapsed}s\n"
            f"  Documents fetched: {len(all_docs)} | Dates: {date_count}\n"
            f"  JSON saved: {json_saved}\n"
            f"{'='*60}"
        )
        return {
            "documents_fetched": len(all_docs),
            "unique_dates": date_count,
            "json_files_saved": json_saved,
            "elapsed_seconds": elapsed,
        }

    def _load_local_docs(self) -> Dict[str, List[Dict]]:
        """Scans RAW_JSON_DIR and loads all JSONs into a docs_by_date dictionary."""
        docs_by_date: Dict[str, List[Dict]] = defaultdict(list)
        if not RAW_JSON_DIR.exists():
            return docs_by_date
            
        for date_dir in RAW_JSON_DIR.iterdir():
            if not date_dir.is_dir():
                continue
            pub_date = date_dir.name
            for json_file in date_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        doc = json.load(f)
                        docs_by_date[pub_date].append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
                    
        return docs_by_date

    def preprocess(self) -> Dict:
        """
        Phase 2: Load local JSONs, build CSVs, run OCR, chunk to ChromaDB, load Postgres.
        """
        start_time = time.time()
        logger.info(
            f"{'='*60}\n"
            f"  PIPELINE START — PREPROCESS\n"
            f"{'='*60}"
        )

        logger.info("[Step 1/5] Loading local JSONs from disk...")
        raw_docs_by_date = self._load_local_docs()
        if not raw_docs_by_date:
            logger.error("No local JSONs found in data directory to preprocess.")
            return {"error": "No data found"}
            
        # Filter documents: Keep only those with an abstract
        docs_by_date: Dict[str, List[Dict]] = defaultdict(list)
        all_docs = []
        for pub_date, docs in raw_docs_by_date.items():
            for doc in docs:
                abstract = doc.get("abstract")
                if abstract and str(abstract).strip():
                    docs_by_date[pub_date].append(doc)
                    all_docs.append(doc)

        date_count = len(docs_by_date)
        if not all_docs:
            logger.error("No documents with abstracts found to preprocess.")
            return {"error": "No documents with abstracts"}
            
        logger.info(f"Loaded {len(all_docs)} documents with abstracts across {date_count} dates.")

        # ── 2. Save date-wise CSVs ────────────────────────────────────
        logger.info("[Step 2/5] Converting to CSV (date-wise)...")
        csv_rows = self._save_csv_by_date(docs_by_date)

        # ── 3. Save consolidated CSVs ─────────────────────────────────
        logger.info("[Step 3/5] Saving consolidated CSVs for database load...")
        self._save_consolidated_csv(all_docs)

        # ── 4. Vectorize (OCR + ChromaDB) ─────────────────────────────
        logger.info("[Step 4/5] Running OCR & vectorizing into ChromaDB...")
        total_chunks = self._vectorize_documents(all_docs)

        # ── 5. Load into PostgreSQL ───────────────────────────────────
        logger.info("[Step 5/5] Loading consolidated data into PostgreSQL...")
        db_loaded = self._load_to_postgres()

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"\n{'='*60}\n"
            f"  PREPROCESS COMPLETE in {elapsed}s\n"
            f"  Documents processed: {len(all_docs)} | Dates: {date_count}\n"
            f"  CSV rows: {csv_rows} | Chunks vectorized: {total_chunks}\n"
            f"  PostgreSQL: {'✓' if db_loaded else '✗'}\n"
            f"{'='*60}"
        )
        
        return {
            "documents_loaded": len(all_docs),
            "csv_rows_saved": csv_rows,
            "chunks_vectorized": total_chunks,
            "postgres_loaded": db_loaded,
            "elapsed_seconds": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Federal Register Data Pipeline")
    parser.add_argument("--action", type=str, choices=["download", "preprocess", "all"], default="download", 
                        help="Action to perform: 'download' fetches JSONs, 'preprocess' does OCR/CSVs/DB, 'all' does both.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--agency", type=str, default=None, help="Filter by agency (e.g. 'Environmental Protection Agency')")
    parser.add_argument("--president", type=str, default=None, help="Filter by president (e.g. 'donald-trump')")
    parser.add_argument("--output-dir", type=str, default="./data", help="Base output directory for JSON/CSV")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Override global paths if output_dir is provided
    global DATA_DIR, RAW_JSON_DIR, PROCESSED_CSV_DIR, CONSOLIDATED_DIR
    DATA_DIR = Path(args.output_dir)
    RAW_JSON_DIR = DATA_DIR / "raw" / "json"
    PROCESSED_CSV_DIR = DATA_DIR / "processed" / "csv"
    CONSOLIDATED_DIR = DATA_DIR / "consolidated"

    # Initialize ChromaDB retrieval agent for vectorization
    from app.agents.retrieval import RetrievalAgent

    retrieval = RetrievalAgent()
    pipeline = FederalRegisterPipeline(retrieval_agent=retrieval)

    # Execute action
    result = {}
    if args.action in ["download", "all"]:
        result["download"] = pipeline.download(
            start_date=args.start_date,
            end_date=args.end_date,
            agency=args.agency, 
            president=args.president
        )
        
    if args.action in ["preprocess", "all"]:
        result["preprocess"] = pipeline.preprocess()

    print("\n── Pipeline Result ──")
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
