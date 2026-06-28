import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
from .ingestion.federal_api import FederalRegisterPipeline
from .agents.retrieval import RetrievalAgent

logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Scheduler Triggered: Running Federal Register Ingestion Pipeline...")
    try:
        retrieval = RetrievalAgent()
        pipeline = FederalRegisterPipeline(retrieval_agent=retrieval)
        result = pipeline.ingest(pages=10, per_page=50)
        logger.info(f"Scheduler: Ingestion Pipeline Completed Successfully. {result}")
    except Exception as e:
        logger.error(f"Scheduler: Ingestion Pipeline Failed: {e}")

async def start_scheduler():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_pipeline,
        CronTrigger(hour=2, minute=0),
        id="daily_federal_register_ingestion",
        name="Daily ingestion of Federal Register documents at 2 AM",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started. Daily ingestion scheduled for 2:00 AM.")
    
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        asyncio.run(start_scheduler())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
