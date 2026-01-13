import os
import asyncio
import logging
import uvicorn
import datetime
from fastapi import FastAPI, BackgroundTasks, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import glob

from models import SessionLocal, init_db, ProcessedRepo, ProcessingStatus
from github_client import GitHubMonitor
from feishu_client import FeishuService
from mcp_client import DeepWikiMCPClient
from deepwiki_indexer import DeepWikiIndexer
from rag_refine import RAGRefiner, Config
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_deepwiki.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize DB
init_db()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Service Instances
github_monitor = None
feishu_service = None
rag_refiner = None
mcp_client = None
deepwiki_indexer = None
templates = Jinja2Templates(directory="/www/wwwroot/mcp_deepwiki/templates")

config = Config()

async def process_repo_workflow(db: Session, repo_data: dict):
    repo_id = str(repo_data["id"])
    repo_name = repo_data["full_name"]
    github_url = repo_data["html_url"]
    
    db_repo = db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first()
    
    if db_repo:
        # Always update timestamp to show we checked it
        db_repo.updated_at = datetime.datetime.utcnow()
        db.commit()
        
    if db_repo and (db_repo.status == ProcessingStatus.COMPLETED or db_repo.status == ProcessingStatus.PROCESSING):
        return
        
    if not db_repo:
        db_repo = ProcessedRepo(
            repo_id=repo_id,
            repo_name=repo_name,
            repo_url=github_url,
            description=repo_data.get("description"),
            status=ProcessingStatus.PROCESSING
        )
        db.add(db_repo)
    else:
        db_repo.status = ProcessingStatus.PROCESSING
    db.commit()
    db.refresh(db_repo)
    
    try:
        logger.info(f"Starting RAG workflow for {repo_name}...")
        
        safe_name = repo_name.replace("/", "_")
        base_dir = f"/www/wwwroot/mcp_deepwiki/output/{safe_name}"
        
        # 1. Fetch from DeepWiki MCP if data is missing
        if not os.path.exists(base_dir) or not glob.glob(os.path.join(base_dir, "*Overview.md")):
            logger.info(f"Data missing for {repo_name}, fetching from MCP...")
            try:
                await mcp_client.fetch_and_save(repo_name)
            except Exception as e:
                logger.warning(f"Failed to fetch data from MCP for {repo_name}: {e}. Attempting to trigger indexing...")
                # If MCP fails, it might be unindexed. Trigger indexing request.
                success = await deepwiki_indexer.request_indexing(repo_name)
                if success:
                    # Mark as failed but with a specific message for retry logic if needed
                    db_repo.status = ProcessingStatus.FAILED
                    db_repo.error_message = f"Indexing requested for {repo_name}. Retrying in 10 minutes..."
                    db.commit()
                    
                    logger.info(f"Indexing triggered for {repo_name}. Waiting 10 minutes for retry...")
                    await asyncio.sleep(600) # Wait 10 minutes
                    
                    # Try fetching again after wait
                    try:
                        await mcp_client.fetch_and_save(repo_name)
                    except Exception as retry_e:
                        raise Exception(f"MCP fetch failed after 10m wait: {retry_e}")
                else:
                    raise Exception(f"MCP fetch failed and indexing request also failed: {e}")

        if not os.path.exists(base_dir):
            raise Exception(f"Repo data folder still not found after MCP fetch: {base_dir}")

        # Find Overview file
        overview_files = glob.glob(os.path.join(base_dir, "*Overview.md"))
        if not overview_files:
            raise Exception("No Overview.md found")
        target_file = overview_files[0]

        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. RAG Refine
        # Generate Draft
        draft = await rag_refiner.generate_draft(content)
        
        # Select documents
        candidate_files = [os.path.basename(p) for p in glob.glob(os.path.join(base_dir, "*.md")) if os.path.abspath(p) != os.path.abspath(target_file)]
        selected_files = await rag_refiner.select_documents_for_rag(draft, candidate_files)
        
        # Build Knowledge Base
        await rag_refiner.build_knowledge_base(base_dir, target_file, selected_files)
        
        # Final Expand
        final_content = await rag_refiner.expand_with_rag(draft)
        
        # 3. Upload to Feishu
        title = f"{repo_name} RAG Refined"
        if not db_repo.feishu_doc_token:
            doc_token = await feishu_service.create_node(title=title)
            if doc_token:
                db_repo.feishu_doc_token = doc_token
                db.commit()
        else:
            doc_token = db_repo.feishu_doc_token

        if doc_token:
            # Note: update_document_content currently appends content.
            # In a production scenario, you might want to clear existing blocks first.
            await feishu_service.update_document_content(doc_token, final_content)
            
            # 4. Notify
            await feishu_service.send_card_notification(
                title=f"RAG Refined Wiki: {repo_name}",
                summary=repo_data.get("description") or "Documentation optimized via RAG workflow.",
                url=f"https://feishu.cn/docx/{doc_token}"
            )
            # Add plain text webhook notification
            await feishu_service.send_webhook_notification(repo_name, doc_token)
        
        db_repo.status = ProcessingStatus.COMPLETED
    except Exception as e:
        logger.error(f"Error: {e}")
        db_repo.status = ProcessingStatus.FAILED
        db_repo.error_message = str(e)
    
    db.commit()

async def sync_task(sync_all: bool = False, silent: bool = False):
    if not silent:
        logger.info(f"Starting sync task (sync_all={sync_all})")
    db = SessionLocal()
    try:
        stars = await github_monitor.fetch_recent_stars(limit=10)
        for star in stars:
            await process_repo_workflow(db, star)
    finally:
        db.close()
        if not silent:
            logger.info("Sync task finished")

# Background Scheduler
async def scheduler_loop():
    while True:
        await asyncio.sleep(30) # Run every 30 seconds
        await sync_task(sync_all=False, silent=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global github_monitor, feishu_service, rag_refiner, mcp_client, deepwiki_indexer
    github_monitor = GitHubMonitor(os.getenv("GITHUB_TOKEN"))
    feishu_service = FeishuService(
        os.getenv("FEISHU_APP_ID"),
        os.getenv("FEISHU_APP_SECRET"),
        os.getenv("FEISHU_SPACE_ID"),
        os.getenv("FEISHU_WEBHOOK_URL")
    )
    rag_refiner = RAGRefiner()
    mcp_client = DeepWikiMCPClient()
    deepwiki_indexer = DeepWikiIndexer(os.getenv("USER_EMAIL"))

    asyncio.create_task(scheduler_loop())

    # Initialize DB with historical stars on first run
    db = SessionLocal()
    try:
        if db.query(ProcessedRepo).count() == 0:
            logger.info("First run: Initializing database with all historical star repositories...")
            stars = await github_monitor.fetch_all_stars()
            for star in stars:
                repo_id = str(star["id"])
                if not db.query(ProcessedRepo).filter(ProcessedRepo.repo_id == repo_id).first():
                    # Check if we already have the output folder for this repo
                    safe_name = star["full_name"].replace("/", "_")
                    base_dir = f"/www/wwwroot/mcp_deepwiki/output/{safe_name}"
                    
                    status = ProcessingStatus.PENDING
                    # If refined file already exists, mark as completed
                    if os.path.exists(os.path.join(base_dir, "refined", "02_Overview_Refined.md")):
                        status = ProcessingStatus.COMPLETED

                    repo = ProcessedRepo(
                        repo_id=repo_id,
                        repo_name=star["full_name"],
                        repo_url=star["html_url"],
                        description=star.get("description"),
                        status=status
                    )
                    db.add(repo)
            db.commit()
            logger.info(f"DB initialized with {len(stars)} repositories.")
    except Exception as e:
        logger.error(f"DB Init error: {e}")
    finally:
        db.close()

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/repos")
async def get_repos(db: Session = Depends(get_db)):
    return db.query(ProcessedRepo).order_by(ProcessedRepo.updated_at.desc()).all()

@app.post("/trigger")
async def trigger(background_tasks: BackgroundTasks, sync_all: bool = False):
    background_tasks.add_task(sync_task, sync_all)
    return {"status": "triggered"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
