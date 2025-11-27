import os
import json
import base64
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import vertexai
from vertexai import agent_engines
from agent_api import normalize_agent_payload
from google.cloud import storage
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote_plus
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from datetime import datetime, timezone, timedelta
from google.genai import types
from google.cloud import storage
from vertexai import rag
from fastapi import BackgroundTasks
import requests

import uuid
from sqlalchemy import text


from models import (
    Base,
    engine,          
    SessionLocal,    
    ConversationMetadata,
    Document,
    JiraConnection,
    RequirementTrace,
    ConversationHistory,
    TestCaseExport
)

from jira_service import (
    get_valid_connection,
    fetch_jira_projects,
    fetch_jira_requirements,
    create_jira_test_case
)
import secrets
from urllib.parse import urlencode
from fastapi.responses import RedirectResponse

from Master_agent.agent import root_agent

# --- 1. SETUP ---
load_dotenv()

# GCS Client Setup
storage_client = storage.Client()
BUCKET_NAME = os.getenv("BUCKET_NAME") # Ensure BUCKET_NAME is in your .env for uploads

# Agent/Vertex AI Config
AGENT_RESOURCE_ID = os.getenv("AGENT_RESOURCE_ID") # Keep if using remote agent engines elsewhere
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_CLOUD_STAGING_BUCKET = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET") # Needed for vertexai.init

# Jira Config
JIRA_DOMAIN = os.getenv("JIRA_DOMAIN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = "SCRUM"
JIRA_ISSUE_TYPE_NAME = "Task"


# State storage for OAuth (use Redis in production)
oauth_states = {}

# OAuth Config
JIRA_OAUTH_CLIENT_ID = os.getenv("JIRA_OAUTH_CLIENT_ID")
JIRA_OAUTH_CLIENT_SECRET = os.getenv("JIRA_OAUTH_CLIENT_SECRET") 
JIRA_OAUTH_CALLBACK_URL = os.getenv("JIRA_OAUTH_CALLBACK_URL")



RAG_CORPUS_ID = os.getenv("DATA_STORE_ID")
RAG_CORPUS_NAME = f"projects/{GOOGLE_CLOUD_PROJECT}/locations/{GOOGLE_CLOUD_LOCATION}/ragCorpora/{RAG_CORPUS_ID}"

# Vertex AI Init (Check if project/location are loaded)
if GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION and GOOGLE_CLOUD_STAGING_BUCKET:
    vertexai.init(
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        staging_bucket=GOOGLE_CLOUD_STAGING_BUCKET
    )
    print("✅ Vertex AI initialized.")
else:
    print("⚠️ Vertex AI NOT initialized - Missing GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, or GOOGLE_CLOUD_STAGING_BUCKET in .env")

try:
    session_service = DatabaseSessionService(db_url=engine.url)
    
    runner = Runner(
        agent=root_agent,
        app_name="HealthCase AI", # Use a consistent app name
        session_service=session_service,
    )
    print("✅ ADK Runner and Session Service initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize ADK services: {e}")
    # You might want to exit the app if this fails
    # exit(1)

try:
    with engine.connect() as conn:
        # Ensure base tables exist
        Base.metadata.create_all(bind=conn)

        # --- Schema synchronization (for older DBs) ---
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id VARCHAR(255);"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;"))

        # --- Extensions and indexes ---
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            ALTER TABLE vector_embeddings 
            ADD COLUMN IF NOT EXISTS embedding vector(768);
        """))

        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_doc_content_hash ON documents(content_hash);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_doc_status_user ON documents(status, user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_vec_document_id ON vector_embeddings(document_id);"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS gcs_uri VARCHAR(500);"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS rag_file_id VARCHAR(500);"))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_doc_session_active
            ON documents(session_id, is_active)
            WHERE status = 'active';
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_vec_embedding_hnsw
            ON vector_embeddings
            USING hnsw (embedding vector_cosine_ops);
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS requirement_traces (
                id SERIAL PRIMARY KEY,
                requirement_id VARCHAR(50) NOT NULL,
                requirement_text TEXT NOT NULL,
                requirement_type VARCHAR(50),
                category VARCHAR(100),
                compliance_standard VARCHAR(50),
                risk_level VARCHAR(20),
                source_section VARCHAR(200),
                regulatory_refs TEXT[],
                source_document_id VARCHAR(255),
                test_case_ids TEXT[],
                jira_issue_keys TEXT[],
                session_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                status VARCHAR(50) DEFAULT 'extracted',
                coverage_percentage INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_req_session ON requirement_traces(session_id);
            CREATE INDEX IF NOT EXISTS idx_req_id_session ON requirement_traces(requirement_id, session_id);
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id SERIAL PRIMARY KEY,
                app_name VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                content JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_conv_app_user ON conversation_history(app_name, user_id);
        """))

        
        conn.execute(text("""
            ALTER TABLE jira_connections 
            ALTER COLUMN access_token TYPE VARCHAR(3000),
            ALTER COLUMN refresh_token TYPE VARCHAR(3000);
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS testcase_exports (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                session_id VARCHAR(255),
                testcase_id VARCHAR(255),
                title VARCHAR(255),
                risk VARCHAR(255),
                testcase_data JSONB,
                jira_key VARCHAR(255),
                jira_url VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Create Indexes for the new table
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tc_export_user ON testcase_exports(user_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tc_export_session ON testcase_exports(session_id);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tc_export_tc_id ON testcase_exports(testcase_id);"))
        
        conn.execute(text("ALTER TABLE conversation_metadata ADD COLUMN IF NOT EXISTS test_case_count INTEGER;"))
                
        conn.commit()

    print("✅ Database tables and vector extension verified.")
except Exception as e:
    print(f"❌ Failed to initialize database extensions/indexes: {e}")


app = FastAPI(title="HealthCase AI Agent API")


origins = [
    "https://fastapi-agent-frontend-342811635923.us-east4.run.app",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    user_id: str
    session_id: str
    message: str

class TestCasePayload(BaseModel):
    title: str = Field(..., example="User Login with Valid Credentials")
    preconditions: Optional[List[str]] = Field(None)
    steps: Optional[List[str]] = Field(None)
    expected_result: Optional[str] = Field(None, alias="expected") # Frontend sends 'expected'
    
class NewSessionRequest(BaseModel):
    user_id: str

class SendMessageRequest(BaseModel):
    user_id: str
    message: str
    
class RenamePayload(BaseModel):
    new_title: str

# --- 2. SYNC HELPERS ---
def call_vertex_agent(user_id: str, session_id: str, message: str) -> list:
    remote_app = agent_engines.get(AGENT_RESOURCE_ID)
    responses = []
    for event in remote_app.stream_query(
        user_id=user_id, session_id=session_id, message=message
    ):
        responses.append(event)
    return responses

def upload_and_query_agent(contents: bytes, filename: str, user_id: str, session_id: str) -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"corpus/{filename}")
    blob.upload_from_string(contents)
    gs_url = f"gs://{BUCKET_NAME}/corpus/{filename}"

    message_text = f"Please add this document to the Requirements Corpus: {gs_url}"
    agent_response = call_vertex_agent(user_id, session_id, message_text)
    normalized_response = [normalize_agent_payload(event) for event in agent_response]

    return {"gs_url": gs_url, "agent_response": normalized_response}

# --- Jira Helpers ---
def get_jira_auth_header() -> str:
    auth_string = f"{JIRA_EMAIL}:{JIRA_API_TOKEN}"
    auth_bytes = auth_string.encode("ascii")
    return f"Basic {base64.b64encode(auth_bytes).decode('ascii')}"

def format_description_for_jira(data: TestCasePayload) -> dict:
    # ... (function from previous response to format description)
    pass



# --- 3. ENDPOINTS ---
@app.post("/agent/run")
async def run_agent(req: AgentRequest):
    raw_response = await run_in_threadpool(
        call_vertex_agent,
        user_id=req.user_id,
        session_id=req.session_id,
        message=req.message
    )
    normalized_events = [normalize_agent_payload(event) for event in raw_response]
    return {"agentResponse": normalized_events, "receivedAt": "TODO-UTC-Timestamp"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...),
                      user_id: str = Form(...),
                      session_id: str = Form(...)):
    try:
        contents = await file.read()
        result = await run_in_threadpool(
            upload_and_query_agent,
            contents=contents,
            filename=file.filename,
            user_id=user_id,
            session_id=session_id
        )
        return {
            "status": "success",
            "filename": file.filename,
            "gs_url": result["gs_url"],
            "agent_response": result["agent_response"]
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/test-jira-connection")
async def test_jira_connection():
    api_url = f"https://{JIRA_DOMAIN}/rest/api/3/project"
    headers = {"Authorization": get_jira_auth_header(), "Accept": "application/json"}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url, headers=headers)
        response.raise_for_status()
        projects = response.json()
        return {
            "status": "Connection Successful!",
            "projects": [{"name": p.get("name"), "key": p.get("key")} for p in projects]
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- CORRECTED JIRA ENDPOINT ---

# @app.post("/create-jira-test-case")
# async def create_jira_test_case(test_case: TestCasePayload):
#     api_url = f"https://{JIRA_DOMAIN}/rest/api/3/issue"
#     headers = {
#         "Authorization": get_jira_auth_header(),
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#     }
    
#     issue_payload = {
#         "fields": {
#             "project": {"key": JIRA_PROJECT_KEY},
#             "summary": test_case.title,
#             "description": format_description_for_jira(test_case),
#             "issuetype": {"name": JIRA_ISSUE_TYPE_NAME}
#         }
#     }

#     try:
#         async with httpx.AsyncClient() as client:
#             response = await client.post(api_url, headers=headers, json=issue_payload)
#         response.raise_for_status()
#         created_issue = response.json()
#         issue_url = f"https://{JIRA_DOMAIN}/browse/{created_issue['key']}"
#         return {
#             "status": "Issue created successfully!",
#             "issue_key": created_issue['key'],
#             "url": issue_url
#         }
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=e.response.status_code, detail=e.response.json())
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
    
# main.py

@app.post("/new-session")
async def create_new_session(req: NewSessionRequest):
    """Creates a new chat session and ensures user_id is in ADK state."""
    db = SessionLocal()
    try:
        # Include user_id in the state passed to create_session
        initial_state = {
            "user_name": req.user_id,
            "user_id": req.user_id, # Add user_id here for the tool
            # Add any other initial state needed by your agent/tools
        }
        print(f"Creating session for user {req.user_id} with initial state: {initial_state}")

        # 1. Create the ADK session with the initial state containing user_id
        new_adk_session = await session_service.create_session(
            app_name=runner.app_name,
            user_id=req.user_id,
            state=initial_state, # Pass the state including user_id
        )
        new_session_id = new_adk_session.id
        print(f"ADK session created with ID: {new_session_id}")

        # --- NO NEED TO UPDATE STATE HERE ---
        # The session_id will be available via tool_context later

        # 2. Save conversation metadata (for listing sessions in UI)
        default_title = "New Conversation"
        new_metadata = ConversationMetadata(
            session_id=new_session_id,
            user_id=req.user_id,
            title=default_title,
            updated_at=datetime.now(timezone.utc)
        )
        db.add(new_metadata)
        db.commit()

        print(f"Created new session metadata: {new_session_id} for user: {req.user_id}")
        return {"session_id": new_session_id, "title": default_title}
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating new session: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")
    finally:
        db.close()
    
@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, req: SendMessageRequest):
    """Sends a message to an existing session and gets the agent's response."""
    db = SessionLocal()
    try:
        content = types.Content(role="user", parts=[types.Part(text=req.message)])
        final_response_text = ""
        
        new_conversation = ConversationHistory(
            app_name=runner.app_name,
            user_id=req.user_id,
            session_id=session_id,
            content={"role": "user", "text": req.message, "aggregated_testcases": [], }
        )
        db.add(new_conversation)
        
        async for event in runner.run_async(
            user_id=req.user_id, session_id=session_id, new_message=content
        ):
            
        #     # if event.actions and event.actions.escalate and event.actions.state_delta and hasattr(event.actions.state_delta, "aggregated_testcases"):
        #     #     final_response_text = event.actions.state_delta["aggregated_testcases"]
        #     #     print("Final response received:\n",final_response_text)
            pass
        
        session = await session_service.get_session(
            app_name=runner.app_name, user_id=req.user_id, session_id=session_id
        )
        agent_state = session.state

        # print("Agent final state:", agent_state)

        # Use .get() with default values - much cleaner!
        final_summary = agent_state.get("final_summary") or "Agent was unable to process the request. Please try again."

        aggregated_testcases = agent_state.get("aggregated_testcases") or [
            {"testcase_id": "N/A", "Testcase Title": "No test cases generated.", 
             "testcases":[],"compliance_ids":[]}
        ]
        
        new_conversation = ConversationHistory(
            app_name=runner.app_name,
            user_id=req.user_id,
            session_id=session_id,
            content={"role": "assistant", "text": final_summary, "aggregated_testcases": aggregated_testcases, }
        )
        
        db.add(new_conversation)
        db.commit()
        
        # --- NEW: Update our metadata table ---
        updated_title = None
        session_metadata = db.query(ConversationMetadata).filter(
            ConversationMetadata.session_id == session_id,
            ConversationMetadata.user_id == req.user_id
        ).first()

        if session_metadata:
            session_metadata.updated_at = datetime.now(timezone.utc)
            if session_metadata.title == "New Conversation":
                new_title = req.message[:50]
                session_metadata.title = new_title
                updated_title = new_title
            
            db.commit()
        
        return {"role": "assistant", "text": final_summary, "updated_title": updated_title, "aggregated_testcases": aggregated_testcases}
    except Exception as e:
        print(f"Error listing sessions for user : {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    """Lists all existing sessions for a user."""
    db = SessionLocal()
    try:
        sessions_metadata = db.query(ConversationMetadata).filter(
            ConversationMetadata.user_id == user_id
        ).order_by(ConversationMetadata.updated_at.desc()).all()
        
        formatted_sessions = []
        updates_needed = False

        for s in sessions_metadata:
            count = s.test_case_count
            
            # 🟢 FALLBACK / SELF-HEALING LOGIC
            # If count is missing (NULL), calculate it now using the fixed helper!
            if count is None:
                history = db.query(ConversationHistory).filter(
                    ConversationHistory.session_id == s.session_id
                ).all()
                
                max_count = 0
                for msg in history:
                    content = msg.content
                    if isinstance(content, dict):
                        aggr = content.get("aggregated_testcases", [])
                        # This now calls the fixed "Sets" counter
                        current = calculate_test_case_count(aggr)
                        if current > max_count:
                            max_count = current
                
                count = max_count
                s.test_case_count = count
                updates_needed = True

            formatted_sessions.append({
                "id": s.session_id, 
                "title": s.title, 
                "updatedAt": s.updated_at.isoformat(),
                "test_case_count": count or 0 
            })
        
        if updates_needed:
            db.commit()

        return {"sessions": formatted_sessions}
    except Exception as e:
        print(f"Error listing sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
        
@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user_id: str): # user_id for security
    """Gets all messages for a specific session."""
    db = SessionLocal()
    try:
        # Fetch all conversation history records
        conversation_records = db.query(ConversationHistory).filter(
            ConversationHistory.app_name == runner.app_name,
            ConversationHistory.session_id == session_id,
            ConversationHistory.user_id == user_id
        ).order_by(ConversationHistory.created_at.asc()).all()
        
        if not conversation_records:
            raise HTTPException(status_code=404, detail="No conversation history found")
        
        # Convert to list of dictionaries
        history_list = [
            {
                "id": record.id,
                "app_name": record.app_name,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "content": record.content,
                "created_at": record.created_at.isoformat() if record.created_at else None
            }
            for record in conversation_records
        ]
        
        return {
            "status": "success",
            "count": len(history_list),
            "conversation_history": history_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    
@app.patch("/sessions/{session_id}/title")
async def rename_session_title(session_id: str, payload: RenamePayload, user_id: str):
    """Updates the title of a specific conversation."""
    db = SessionLocal()
    try:
        session_metadata = db.query(ConversationMetadata).filter(
            ConversationMetadata.session_id == session_id,
            ConversationMetadata.user_id == user_id
        ).first()

        if not session_metadata:
            raise HTTPException(status_code=404, detail="Session not found")

        session_metadata.title = payload.new_title
        session_metadata.updated_at = datetime.now(timezone.utc)
        db.commit()

        return {"status": "success", "new_title": payload.new_title}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        

# ========== RAG DOCUMENT ENDPOINTS ==========

@app.post("/api/rag/upload")
async def upload_document_rag(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: str = Form(...)  # Frontend is sending this
):
    """
    (Simplified) Uploads a document and links it to a session.
    This version just creates the database record so it appears in the UI.
    """
    if not BUCKET_NAME:
         raise HTTPException(status_code=500, detail="GCS Bucket name not configured.")
     
    db = SessionLocal()
    try:
        # 1. Upload to GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        # Create a unique path, maybe including user/session ID
        blob_name = f"user_{user_id}/session_{session_id}/{uuid.uuid4()}_{file.filename}"
        blob = bucket.blob(blob_name)
        
        contents = await file.read()
        blob.upload_from_string(contents, content_type=file.content_type)
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        
        print(f"✅ File uploaded to GCS: {gcs_uri}")
        
        # Create a placeholder document record
        doc_id = str(uuid.uuid4())
        new_doc = Document(
            id=str(doc_id),
            filename=file.filename,
            user_id=user_id,
            session_id=session_id,
            is_active=True,
            status='processing',
            # Simulating some data for the UI
            total_pages=0, 
            chunk_count=0,
            gcs_uri=gcs_uri,
            document_summary="Processing document...",
            rag_file_id=None
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)

        print(f"✅ (Simplified Upload) Saved doc {new_doc.id} to session {session_id}")
        
        # 3. Import to RAG Engine (async in background)
        background_tasks.add_task(
            import_to_rag_engine,
            doc_id=str(doc_id),
            gcs_uri=gcs_uri,
            user_id=user_id,
            session_id=session_id,
            filename=file.filename
        )

        return {
            "status": "success",
            "document_id": new_doc.id,
            "filename": new_doc.filename,
            "gcs_uri": gcs_uri,
            "message": "Document uploaded and is processing."
        }
        # --- END OF SIMPLIFIED VERSION ---
    
    except Exception as e:
        db.rollback() # Rollback DB changes if GCS upload or DB save fails
        print(f"❌ RAG Upload Error: {e}")
        # Attempt to delete the GCS file if DB save failed
        if blob:
            try:
                print(f"   -> Attempting to clean up GCS blob: {blob.name}")
                blob.delete()
                print(f"   -> GCS blob deleted.")
            except Exception as delete_e:
                print(f"   -> Failed to delete GCS blob {blob.name}: {delete_e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")
    finally:
        db.close()
        
def import_to_rag_engine(
    doc_id: str,
    gcs_uri: str,
    user_id: str,
    session_id: str,
    filename: str
):
    """
    Background task to import document to RAG Engine
    """
    db = SessionLocal()
    try:
        print(f"📥 Starting RAG import for {doc_id}")
        
        # Import to RAG Engine with chunking config
        response = rag.import_files(
            corpus_name=RAG_CORPUS_NAME,
            paths=[gcs_uri],
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=512,      # Adjust based on your needs
                    chunk_overlap=100
                )
            ),
            max_embedding_requests_per_min=900
        )
        
        # Get the RAG file ID
        rag_file_id = None
        if response.imported_rag_files_count > 0:
            try:
                # List files and find the one we just imported
                files = list(rag.list_files(corpus_name=RAG_CORPUS_NAME))
                
                # Find by GCS URI or filename (most recently added)
                for file in reversed(files):  
                    if filename in file.display_name:
                        rag_file_id = file.name
                        print(f"✅ Found RAG file: {rag_file_id}")
                        break
                
                # If not found by display name, use the last file
                if not rag_file_id and files:
                    rag_file_id = files[-1].name
                    print(f"⚠️ Using last file as fallback: {rag_file_id}")
                    
            except Exception as list_error:
                print(f"⚠️ Could not list files: {list_error}")
                # Fallback: construct expected ID
                rag_file_id = f"{RAG_CORPUS_NAME}/ragFiles/{doc_id}"
        
        if not rag_file_id:
            raise Exception("Failed to get RAG file ID after import")
        
        print(f"✅ RAG import complete: {rag_file_id}")
        
        # Update document record
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.rag_file_id = rag_file_id
            doc.status = 'active'
            doc.document_summary = f"Document processed successfully. Ready for querying."
            doc.chunk_count = 1  # You can calculate actual chunks if needed
            db.commit()
            print(f"✅ Updated document {doc_id} with RAG file ID")
    
    except Exception as e:
        print(f"❌ RAG import failed for {doc_id}: {e}")
        # Update document status to failed
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = 'failed'
            doc.document_summary = f"Failed to process: {str(e)}"
            db.commit()
        import traceback
        traceback.print_exc()
    finally:
        db.close()


@app.get("/api/rag/documents/session/{session_id}")
async def get_session_documents(session_id: str, user_id: str):
    """
    Gets all documents associated with a specific session for a user.
    (This is called by DocumentManager.tsx)
    """
    db = SessionLocal()
    try:
        # Query using SQLAlchemy ORM (safer than raw SQL)
        docs = db.query(
            Document.id,
            Document.filename,
            Document.chunk_count,
            Document.total_pages,
            Document.document_summary,
            Document.upload_date,
            Document.is_active
        ).filter(
            Document.session_id == session_id,
            Document.user_id == user_id,
            Document.status == 'active'
        ).order_by(Document.upload_date.desc()).all()

        active_count = sum(1 for d in docs if d.is_active)

        return {
            "documents": [
                {
                    "id": str(d.id), # Ensure ID is string
                    "filename": d.filename,
                    "chunk_count": d.chunk_count,
                    "total_pages": d.total_pages,
                    "summary": d.document_summary,
                    "uploaded": d.upload_date.isoformat() if d.upload_date else None,
                    "is_active": d.is_active
                }
                for d in docs
            ],
            "total": len(docs),
            "active_count": active_count
        }
    except Exception as e:
        print(f"❌ Get Session Docs Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.patch("/api/rag/documents/{document_id}/toggle")
async def toggle_document_active(
    document_id: str,
    user_id: str = Form(...), # Read user_id from form body
    is_active: bool = Form(...) # Read new status from form body
):
    """
    Toggles a document's active status.
    (This is called by DocumentManager.tsx via api.js)
    """
    db = SessionLocal()
    try:
        # Verify ownership
        doc = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id,
            Document.status == 'active'
        ).first()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or not authorized")

        # Set the new status from the request
        doc.is_active = is_active
        db.commit()

        print(f"✅ Toggled doc {doc.id} for user {user_id} to {is_active}")

        return {
            "status": "success",
            "document_id": doc.id,
            "is_active": doc.is_active
        }
    except Exception as e:
        db.rollback()
        print(f"❌ Toggle Doc Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
        
# ============================================================================
# JIRA OAUTH ENDPOINTS
# ============================================================================

@app.get("/api/jira/connect")
async def jira_connect(user_id: str):
    """Initiate OAuth flow."""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = user_id
    
    params = {
        "audience": "api.atlassian.com",
        "client_id": JIRA_OAUTH_CLIENT_ID,
        "scope": (
            "read:jira-work read:jira-user write:jira-work "
            "read:jira-software offline_access"
        ),
        "redirect_uri": JIRA_OAUTH_CALLBACK_URL,
        "state": state,
        "response_type": "code",
        "prompt": "consent"
    }
    
    auth_url = f"https://auth.atlassian.com/authorize?{urlencode(params)}"
    return {"authorization_url": auth_url}


@app.get("/api/jira/callback")
async def jira_callback(code: str, state: str):
    """Handle OAuth callback."""
    user_id = oauth_states.get(state)
    if not user_id:
        raise HTTPException(400, "Invalid state")
    
    # Exchange code for tokens
    token_url = "https://auth.atlassian.com/oauth/token"
    payload = {
        "grant_type": "authorization_code",
        "client_id": JIRA_OAUTH_CLIENT_ID,
        "client_secret": JIRA_OAUTH_CLIENT_SECRET,
        "code": code,
        "redirect_uri": JIRA_OAUTH_CALLBACK_URL
    }
    
    print("CLIENT_ID present:", bool(JIRA_OAUTH_CLIENT_ID))
    print("CLIENT_SECRET present:", bool(JIRA_OAUTH_CLIENT_SECRET))
    print("CALLBACK URL:", JIRA_OAUTH_CALLBACK_URL)

    print("Token request payload (safe):", {
        "grant_type": payload["grant_type"],
        "client_id_present": bool(payload["client_id"]),
        "redirect_uri": payload["redirect_uri"],
        "code_present": bool(payload["code"]),
    })

    response = requests.post(token_url, json=payload)
    print("Token response status:", response.status_code)
    print("Token response body:", response.text)
    response.raise_for_status()

    
    # response = requests.post(token_url, json=payload)
    # response.raise_for_status()
    tokens = response.json()
    
    print("tokens received:", {k: (v if k != "access_token" else "****") for k, v in tokens.items()})
    
    # Get Jira instance
    resources_url = "https://api.atlassian.com/oauth/token/accessible-resources"
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    resources_resp = requests.get(resources_url, headers=headers)
    resources = resources_resp.json()
    
    if not resources:
        raise HTTPException(400, "No Jira instances")
    
    jira_resource = resources[0]
    
    # Save to database
    db = SessionLocal()
    try:
        existing = db.query(JiraConnection).filter(
            JiraConnection.user_id == user_id
        ).first()
        
        if existing:
            existing.access_token = tokens["access_token"]
            existing.refresh_token = tokens.get("refresh_token")
            existing.token_expires_at = datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))
            existing.jira_cloud_id = jira_resource["id"]
            existing.jira_base_url = jira_resource["url"]
            existing.is_active = True
        else:
            conn = JiraConnection(
                user_id=user_id,
                jira_cloud_id=jira_resource["id"],
                jira_base_url=jira_resource["url"],
                access_token=tokens["access_token"],
                refresh_token=tokens.get("refresh_token"),
                token_expires_at=datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))
            )
            db.add(conn)
        
        db.commit()
        del oauth_states[state]
        
        # Redirect to frontend
        return RedirectResponse(url="http://localhost:5173/jira-connected?success=true")
    finally:
        db.close()


class JiraCallbackPayload(BaseModel):
    user_id: str
    code: str
    
    
    
@app.post("/api/jira/callback")
async def jira_callback_post(payload: JiraCallbackPayload):
    """
    Receives the auth code from the Frontend and exchanges it for a token.
    """
    # 1. Exchange code for tokens
    token_url = "https://auth.atlassian.com/oauth/token"
    token_payload = {
        "grant_type": "authorization_code",
        "client_id": JIRA_OAUTH_CLIENT_ID,
        "client_secret": JIRA_OAUTH_CLIENT_SECRET,
        "code": payload.code,
        "redirect_uri": JIRA_OAUTH_CALLBACK_URL # Ensure this matches what you set in Jira Console
    }

    try:
        # We use requests or httpx here. Since you used requests in the GET route:
        response = requests.post(token_url, json=token_payload)
        response.raise_for_status()
        tokens = response.json()

        # 2. Get Jira Instance ID (Cloud ID)
        resources_url = "https://api.atlassian.com/oauth/token/accessible-resources"
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        resources_resp = requests.get(resources_url, headers=headers)
        resources_resp.raise_for_status()
        resources = resources_resp.json()

        if not resources:
            raise HTTPException(status_code=400, detail="No Jira instances found for this user.")
        
        jira_resource = resources[0] # Taking the first available Jira site

        # 3. Save to Database
        db = SessionLocal()
        try:
            # Check if user already has a connection
            existing = db.query(JiraConnection).filter(
                JiraConnection.user_id == payload.user_id
            ).first()

            expires_in = tokens.get("expires_in", 3600)
            
            if existing:
                existing.access_token = tokens["access_token"]
                existing.refresh_token = tokens.get("refresh_token")
                existing.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                existing.jira_cloud_id = jira_resource["id"]
                existing.jira_base_url = jira_resource["url"]
                existing.is_active = True
            else:
                conn = JiraConnection(
                    user_id=payload.user_id,
                    jira_cloud_id=jira_resource["id"],
                    jira_base_url=jira_resource["url"],
                    access_token=tokens["access_token"],
                    refresh_token=tokens.get("refresh_token"),
                    token_expires_at=datetime.utcnow() + timedelta(seconds=expires_in)
                )
                db.add(conn)
            
            db.commit()
            return {"status": "success", "connected": True}
            
        except Exception as db_e:
            db.rollback()
            raise db_e
        finally:
            db.close()

    except Exception as e:
        print(f"Error in Jira Callback: {e}")
        # Return 500 so frontend knows it failed
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jira/status")
async def jira_status(user_id: str):
    """Check connection status."""
    conn = get_valid_connection(user_id)
    if conn:
        return {
            "connected": True,
            "jira_url": conn.jira_base_url,
            "expires_at": conn.token_expires_at.isoformat()
        }
    return {"connected": False}


@app.delete("/api/jira/disconnect")
async def jira_disconnect(user_id: str):
    """Disconnect Jira."""
    db = SessionLocal()
    try:
        db.query(JiraConnection).filter(
            JiraConnection.user_id == user_id
        ).update({"is_active": False})
        db.commit()
        return {"status": "disconnected"}
    finally:
        db.close()


# ============================================================================
# JIRA OPERATIONS (REPLACE YOUR HARDCODED ONES)
# ============================================================================

@app.get("/api/jira/projects")
async def get_projects(user_id: str):
    """Get projects (OAuth)."""
    return fetch_jira_projects(user_id)


@app.post("/api/jira/fetch-requirements")
async def fetch_requirements(data: dict):
    """Fetch requirements (OAuth)."""
    return fetch_jira_requirements(data["user_id"], data["project_key"])


@app.post("/api/jira/create-jira-test-case")
async def create_test_case_oauth(data: dict):
    """Create test case (OAuth)."""
    # Expected data: { user_id, project_key, test_case: {...}, session_id?, testcase_id?, requirement_key? }
    db = SessionLocal()
    try:
        user_id = data.get("user_id")
        project_key = data.get("project_key")
        test_case = data.get("test_case")
        requirement_key = data.get("requirement_key")
        session_id = data.get("session_id") or (test_case or {}).get("session_id")
        testcase_id = data.get("testcase_id") or (test_case or {}).get("id")

        # Call Jira service to create the issue
        result = create_jira_test_case(
            user_id=user_id,
            project_key=project_key,
            test_case=test_case or {},
            requirement_key=requirement_key
        )

        # If creation succeeded, persist an export record so UI can show permanent success
        if isinstance(result, dict) and result.get("status") == "success":
            try:
                export = TestCaseExport(
                    user_id=user_id,
                    session_id=session_id,
                    testcase_id=str(testcase_id) if testcase_id is not None else None,
                    title=(test_case or {}).get("title"),
                    risk=(test_case or {}).get("risk"),
                    testcase_data=test_case or {},
                    jira_key=result.get("jira_key"),
                    jira_url=result.get("jira_url")
                )
                db.add(export)
                db.commit()
                # Return enriched response including saved export id
                return {**result, "export_id": export.id}
            except Exception as db_e:
                db.rollback()
                # Log but still return the jira result so frontend can show link
                print(f"❌ Failed to save TestCaseExport: {db_e}")
                return {**result, "export_id": None, "warning": "failed_to_persist_export"}

        # Propagate errors from Jira service
        return result
    finally:
        db.close()
    
# main.py - SIMPLE IMPORT ENDPOINT

# Modify the existing /api/jira/import-requirements endpoint

@app.post("/api/jira/import-requirements")
async def import_jira_requirements(background_tasks: BackgroundTasks, data: dict):
    """
    Import selected Jira requirements.
    Saves to database + uploads to RAG corpus.
    """
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    requirements = data.get("requirements", [])
    overwrite = data.get("overwrite", False)  # NEW: Allow overwrite flag
    
    if not all([session_id, user_id, requirements]):
        raise HTTPException(400, "Missing required fields")
    
    db = SessionLocal()
    try:
        imported_count = 0
        updated_count = 0
        
        # 1. Save to database for tracking/UI
        for req_data in requirements:
            existing = db.query(RequirementTrace).filter(
                RequirementTrace.requirement_id == req_data.get("id"),
                RequirementTrace.session_id == session_id
            ).first()
            
            if existing:
                if overwrite:
                    # Update existing requirement
                    existing.requirement_text = req_data.get("text")
                    existing.requirement_type = req_data.get("type", "functional")
                    existing.risk_level = req_data.get("risk_level", "medium")
                    existing.compliance_standard = req_data.get("compliance_standard", "None")
                    existing.updated_at = datetime.now()
                    updated_count += 1
                else:
                    # Skip if not overwriting
                    continue
            else:
                # Create new requirement
                requirement = RequirementTrace(
                    requirement_id=req_data.get("id"),
                    requirement_text=req_data.get("text"),
                    requirement_type=req_data.get("type", "functional"),
                    category="from_jira",
                    compliance_standard=req_data.get("compliance_standard", "None"),
                    risk_level=req_data.get("risk_level", "medium"),
                    source_section=f"Jira: {req_data.get('jira_key')}",
                    regulatory_refs=req_data.get("regulatory_refs", []),
                    jira_issue_keys=[req_data.get("jira_key")],
                    session_id=session_id,
                    user_id=user_id,
                    status='imported'
                )
                db.add(requirement)
                imported_count += 1
        
        db.commit()
        
        # 2. Upload to RAG corpus in background (only if new imports)
        if imported_count > 0:
            background_tasks.add_task(
                upload_requirements_to_rag,
                requirements=requirements,
                session_id=session_id,
                user_id=user_id
            )
        
        message = f"Imported {imported_count} new requirements"
        if updated_count > 0:
            message += f", updated {updated_count} existing requirements"
        
        return {
            "status": "success",
            "imported": imported_count,
            "updated": updated_count,
            "message": message
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))
    finally:
        db.close()


@app.get("/api/jira/exports")
async def get_jira_exports(
    user_id: str = Query(...),
    session_id: str = Query(None)
):
    """Retrieve all Jira exports for a user (optionally filtered by session)."""
    db = SessionLocal()
    try:
        # CORRECTION 1: Use correct model attribute names (user_id, not userid)
        query = db.query(TestCaseExport).filter(
            TestCaseExport.user_id == user_id
        )
        
        if session_id:
            query = query.filter(TestCaseExport.session_id == session_id)
        
        # CORRECTION 2: Use correct created_at
        exports = query.order_by(TestCaseExport.created_at.desc()).all()
        
        return {
            "exports": [
                {
                    "id": e.id,
                    "testcase_id": e.testcase_id,   # Fixed
                    "title": e.title,
                    "risk": e.risk,
                    "jira_key": e.jira_key,         # Fixed
                    "jira_url": e.jira_url,         # Fixed
                    "testcase_data": e.testcase_data, # Fixed
                    "created_at": e.created_at.isoformat() if e.created_at else None # Fixed
                } for e in exports
            ],
            "total": len(exports)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()



# --- Analytics endpoints ---
@app.get("/api/analytics/overview")
async def analytics_overview(user_id: str):
    """
    Return high-level analytics. 
    Auto-updates 'test_case_count' in metadata if missing.
    """
    db = SessionLocal()
    try:
        # 1. SELF-HEALING: Find sessions with NULL counts and fix them
        # This ensures the dashboard numbers are always accurate without manual scripts.
        unprocessed_sessions = db.query(ConversationMetadata).filter(
            ConversationMetadata.user_id == user_id,
            ConversationMetadata.test_case_count == None
        ).all()

        if unprocessed_sessions:
            print(f"🔄 Recalculating stats for {len(unprocessed_sessions)} sessions...")
            for session_meta in unprocessed_sessions:
                # Fetch full history for this session
                history = db.query(ConversationHistory).filter(
                    ConversationHistory.session_id == session_meta.session_id
                ).all()
                
                # Logic: aggregated_testcases is usually cumulative or the latest state.
                # We look for the maximum count found in any message in this session.
                max_count = 0
                for msg in history:
                    content = msg.content
                    if isinstance(content, dict):
                        aggr = content.get("aggregated_testcases", [])
                        current = calculate_test_case_count(aggr)
                        if current > max_count:
                            max_count = current
                
                # Save to Metadata so we never have to calculate this again
                session_meta.test_case_count = max_count
            
            db.commit()


        # 2. Query Analytics (Now fast and accurate)
        total_sessions = db.query(ConversationMetadata).filter(ConversationMetadata.user_id == user_id).count()

        # Sum the column we just ensured is populated
        total_test_cases = db.query(func.sum(ConversationMetadata.test_case_count))\
            .filter(ConversationMetadata.user_id == user_id).scalar() or 0

        # Calculate Average
        avg_test_cases = 0
        if total_sessions > 0:
            avg_test_cases = round(total_test_cases / total_sessions, 1)

        # Other metrics (unchanged)
        total_exports = db.query(TestCaseExport).filter(TestCaseExport.user_id == user_id).count()
        
        # Recent Exports (unchanged)
        recent_exports = (
            db.query(TestCaseExport)
            .filter(TestCaseExport.user_id == user_id)
            .order_by(TestCaseExport.created_at.desc())
            .limit(10)
            .all()
        )
        recent_exports_list = [
            {
                "id": e.id,
                "session_id": e.session_id,
                "testcase_id": e.testcase_id,
                "title": e.title,
                "jira_key": e.jira_key,
                "jira_url": e.jira_url,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_exports
        ]

        # Requirement traces (unchanged)
        req_traces = db.query(RequirementTrace).filter(RequirementTrace.user_id == user_id).count()
        docs_count = db.query(Document).filter(Document.user_id == user_id).count()

        # Exports by session (unchanged)
        exports_by_session = (
            db.query(TestCaseExport.session_id, func.count(TestCaseExport.id).label('count'))
            .filter(TestCaseExport.user_id == user_id)
            .group_by(TestCaseExport.session_id)
            .order_by(text('count DESC'))
            .limit(20)
            .all()
        )
        exports_by_session_list = [{"session_id": s[0], "count": s[1]} for s in exports_by_session]

        return {
            "total_sessions": total_sessions,
            "total_test_cases": total_test_cases,         # ✅ Corrected
            "avg_test_cases_per_session": avg_test_cases, # ✅ Corrected
            "total_exports": total_exports,
            "exports_by_session": exports_by_session_list,
            "recent_exports": recent_exports_list,
            "documents_uploaded": docs_count,
            "requirement_traces": req_traces,
        }

    except Exception as e:
        print(f"❌ Analytics overview error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
        
@app.get("/api/analytics/session/{session_id}")
async def analytics_session_detail(session_id: str, user_id: str):
    """Return analytics for a specific session: messages, exports, docs, traces."""
    db = SessionLocal()
    try:
        # Conversation messages
        messages = (
            db.query(ConversationHistory)
            .filter(ConversationHistory.session_id == session_id, ConversationHistory.user_id == user_id)
            .order_by(ConversationHistory.created_at.asc())
            .all()
        )

        messages_list = [
            {"id": m.id, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in messages
        ]

        # Exports for session
        exports = (
            db.query(TestCaseExport)
            .filter(TestCaseExport.session_id == session_id, TestCaseExport.user_id == user_id)
            .order_by(TestCaseExport.created_at.asc())
            .all()
        )

        exports_list = [
            {"id": e.id, "testcase_id": e.testcase_id, "title": e.title, "jira_key": e.jira_key, "jira_url": e.jira_url, "created_at": e.created_at.isoformat() if e.created_at else None}
            for e in exports
        ]

        # Documents for session
        docs = (
            db.query(Document)
            .filter(Document.session_id == session_id, Document.user_id == user_id)
            .order_by(Document.created_at.asc())
            .all()
        )

        docs_list = [ {"id": d.id, "filename": d.filename, "created_at": d.created_at.isoformat() if d.created_at else None} for d in docs ]

        # Requirement traces for session
        traces = (
            db.query(RequirementTrace)
            .filter(RequirementTrace.session_id == session_id, RequirementTrace.user_id == user_id)
            .order_by(RequirementTrace.created_at.asc())
            .all()
        )

        traces_list = [ {"id": t.id, "requirement_id": t.requirement_id, "coverage": t.coverage_percentage, "created_at": t.created_at.isoformat() if t.created_at else None} for t in traces ]

        return {
            "session_id": session_id,
            "messages": messages_list,
            "exports": exports_list,
            "documents": docs_list,
            "requirement_traces": traces_list,
            "totals": {
                "message_count": len(messages_list),
                "export_count": len(exports_list),
                "document_count": len(docs_list),
                "trace_count": len(traces_list),
            }
        }

    except Exception as e:
        print(f"❌ Analytics session detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


from sqlalchemy import func

@app.get("/api/analytics/exports-timeseries")
async def analytics_exports_timeseries(user_id: str, days: int = 30):
    """Return exports-per-day timeseries for the last `days` days for charts."""
    db = SessionLocal()
    try:
        # Build date series
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days-1)

        # Query counts grouped by date
        rows = (
            db.query(func.date_trunc('day', TestCaseExport.created_at).label('day'), func.count(TestCaseExport.id))
            .filter(TestCaseExport.user_id == user_id, TestCaseExport.created_at >= start_date)
            .group_by('day')
            .order_by('day')
            .all()
        )

        counts_by_day = { r[0].date().isoformat(): r[1] for r in rows }

        # Fill missing dates with zero
        series = []
        for i in range(days):
            d = (start_date + timedelta(days=i)).date().isoformat()
            series.append({"date": d, "count": counts_by_day.get(d, 0)})

        return {"series": series}

    except Exception as e:
        print(f"❌ Exports timeseries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()



def upload_requirements_to_rag(requirements: list, session_id: str, user_id: str):
    """Background task to upload requirements to RAG corpus."""
    try:
        
        doc_content = "# Imported Requirements from Jira\n\n"
        
        for req in requirements:
            doc_content += f"## {req.get('id')}: {req.get('text')}\n\n"
            doc_content += f"**Jira Key:** {req.get('jira_key')}\n"
            doc_content += f"**Type:** {req.get('type', 'Functional')}\n"
            doc_content += f"**Risk Level:** {req.get('risk_level', 'Medium').upper()}\n"
            doc_content += f"**Compliance:** {req.get('compliance_standard', 'None')}\n"
            
            if req.get('description'):
                doc_content += f"\n**Description:**\n{req.get('description')}\n"
            
            doc_content += "\n---\n\n"
        
        # Upload to GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"user_{user_id}/session_{session_id}/jira_requirements_{uuid.uuid4()}.txt"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(doc_content, content_type="text/plain")
        gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
        
        # Import to RAG corpus
        rag.import_files(
            corpus_name=RAG_CORPUS_NAME,
            paths=[gcs_uri],
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=512,
                    chunk_overlap=100
                )
            )
        )
        
        print(f"✅ Uploaded {len(requirements)} requirements to RAG corpus")
        
    except Exception as e:
        print(f"❌ Failed to upload requirements to RAG: {e}")
        import traceback
        traceback.print_exc()

# main.py - ADD THIS ENDPOINT

@app.get("/api/requirements/session/{session_id}")
async def get_session_requirements_ui(session_id: str, user_id: str):
    """Get all requirements for session for UI display."""
    db = SessionLocal()
    try:
        from models import RequirementTrace
        
        requirements = db.query(RequirementTrace).filter(
            RequirementTrace.session_id == session_id,
            RequirementTrace.user_id == user_id
        ).all()
        
        return {
            "requirements": [
                {
                    "id": str(req.id),  # Use the database ID, not requirement_id
                    "requirement_id": req.requirement_id,  # Keep for reference
                    "text": req.requirement_text,
                    "type": req.requirement_type,
                    "risk_level": req.risk_level,
                    "compliance_standard": req.compliance_standard,
                    "jira_key": req.jira_issue_keys[0] if req.jira_issue_keys else None,
                    "status": req.status,
                    "test_case_count": len(req.test_case_ids) if req.test_case_ids else 0,
                }
                for req in requirements
            ],
            "total": len(requirements)
        }
    finally:
        db.close()


# Add after the existing /api/requirements/session/{session_id} endpoint

@app.delete("/api/requirements/{requirement_id}")
async def delete_requirement(
    requirement_id: str,
    user_id: str = Query(..., description="User ID must be provided"),
    session_id: str = Query(..., description="Session ID must be provided"),
):
    """Delete a single requirement and clean up associated data."""
    db = SessionLocal()
    try:
        # FIXED: Query by 'id' column, not 'requirement_id'
        requirement = db.query(RequirementTrace).filter(
            RequirementTrace.id == requirement_id,  # Changed this line
            RequirementTrace.user_id == user_id,
            RequirementTrace.session_id == session_id
        ).first()
        
        if not requirement:
            print(f"❌ Requirement not found: id={requirement_id}, user={user_id}, session={session_id}")
            raise HTTPException(status_code=404, detail="Requirement not found")
        
        # Delete the requirement from database
        req_text = requirement.requirement_text[:50]  # For logging
        db.delete(requirement)
        db.commit()
        
        print(f"✅ Deleted requirement '{req_text}...' (id={requirement_id}) for user {user_id}")
        
        return {
            "status": "success",
            "message": f"Requirement deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Delete requirement error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()



@app.post("/api/jira/check-duplicate-requirements")
async def check_duplicate_requirements(data: dict):
    """Check if requirements already exist before import."""
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    requirement_ids = data.get("requirement_ids", [])
    
    if not all([session_id, user_id, requirement_ids]):
        raise HTTPException(400, "Missing required fields")
    
    db = SessionLocal()
    try:
        # Check which requirements already exist
        existing = db.query(RequirementTrace).filter(
            RequirementTrace.session_id == session_id,
            RequirementTrace.user_id == user_id,
            RequirementTrace.requirement_id.in_(requirement_ids)
        ).all()
        
        existing_ids = [req.requirement_id for req in existing]
        
        return {
            "has_duplicates": len(existing_ids) > 0,
            "existing_requirement_ids": existing_ids,
            "count": len(existing_ids)
        }
    
    finally:
        db.close()
        
        
        
# In main.py, replace the calculate_test_case_count function

def calculate_test_case_count(aggregated_testcases: list) -> int:
    """
    Counts the number of Test Sets (Suites) generated.
    Returns the length of the list (e.g., 2 sets).
    """
    if aggregated_testcases and isinstance(aggregated_testcases, list):
        return len(aggregated_testcases)
    return 0


        
@app.get("/")
async def root():
    return {"message": "HealthCase AI Agent API is running"}




