# models.py
import os
import uuid
from sqlalchemy import create_engine, Column, String, DateTime, text, Integer, JSON, Text, Index, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from urllib.parse import quote_plus
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables to get DB credentials
load_dotenv()

# --- DATABASE CONNECTION SETUP ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PUBLIC_IP = os.getenv("DB_PUBLIC_IP")
DB_NAME = os.getenv("DB_NAME")

# Ensure DB credentials are loaded
if not all([DB_USER, DB_PASSWORD, DB_PUBLIC_IP, DB_NAME]):
    raise ValueError("Database environment variables (DB_USER, DB_PASSWORD, DB_PUBLIC_IP, DB_NAME) are not fully set.")

encoded_password = quote_plus(DB_PASSWORD)
db_url = f"postgresql+psycopg2://{DB_USER}:{encoded_password}@{DB_PUBLIC_IP}/{DB_NAME}"

engine = create_engine(db_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- ALL YOUR TABLE MODELS ---

class ConversationMetadata(Base):
    __tablename__ = 'conversation_metadata'
    session_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

class Document(Base):
    __tablename__ = 'documents'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    content_hash = Column(String(64), unique=True, index=True, nullable=True) # Hash might be added later
    version = Column(Integer, default=1)
    parent_document_id = Column(String(36), nullable=True)
    chunk_count = Column(Integer)
    total_pages = Column(Integer)
    document_summary = Column(Text)
    metadata_json = Column(JSONB)
    upload_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String(20), default='active', index=True) # active, processing, failed, archived
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    gcs_uri = Column(String(1024), nullable=True) # Store the GCS path here
    rag_file_id = Column(String(1024), nullable=True) # Store the GCS path here

class VectorEmbedding(Base):
    __tablename__ = 'vector_embeddings' # Still useful if you add other vector features later

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(36), nullable=False, index=True)
    chunk_index = Column(Integer)
    chunk_type = Column(String(50))
    section_title = Column(String(500))
    subsection_title = Column(String(500))
    text_content = Column(Text)
    # embedding column added via SQL
    page_number = Column(Integer)
    metadata_json = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class DocumentVersion(Base):
    __tablename__ = 'document_versions' # Useful for tracking changes

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_document_id = Column(String(36), nullable=False)
    version_number = Column(Integer, nullable=False)
    filename = Column(String(255))
    content_hash = Column(String(64))
    change_summary = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(255))
    is_active = Column(Boolean, default=True) # Refers to the version, not searchability
    
class RequirementTrace(Base):
    __tablename__ = "requirement_traces"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    requirement_id = Column(String, nullable=False)  # REQ-001
    requirement_text = Column(Text, nullable=False)
    requirement_type = Column(String)  # functional, security, etc.
    category = Column(String)  # authentication, data_security
    compliance_standard = Column(String)  # FDA, IEC 62304, ISO 13485
    risk_level = Column(String)  # high, medium, low
    source_section = Column(String)  # Section 3.2.1
    regulatory_refs = Column(ARRAY(String))  # ["21 CFR Part 11"]
    
    # Traceability
    source_document_id = Column(String)
    test_case_ids = Column(ARRAY(String), default=[])  # [TC-001, TC-002]
    jira_issue_keys = Column(ARRAY(String), default=[])  # [TEST-123]
    
    # Session context
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    
    # Status
    status = Column(String, default='extracted')  # extracted, covered, partial, missing
    coverage_percentage = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class JiraConnection(Base):
    __tablename__ = "jira_connections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Jira instance details
    jira_cloud_id = Column(String(255))
    jira_base_url = Column(String(500))
    
    # OAuth tokens
    access_token = Column(String(3000), nullable=False)  # ✅ Changed from 1000 to 3000
    refresh_token = Column(String(3000))
    token_expires_at = Column(DateTime)
    
    # User info from Jira
    jira_user_email = Column(String(255))
    jira_user_display_name = Column(String(255))
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<JiraConnection(user_id='{self.user_id}', jira_url='{self.jira_base_url}')>"
    
    
class ConversationHistory(Base):
    __tablename__ = 'conversation_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    app_name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255), nullable=False)
    content = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ConversationHistory(id={self.id}, app_name='{self.app_name}', session_id='{self.session_id}')>"
    

class TestCaseExport(Base):
    __tablename__ = "testcase_exports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    session_id = Column(String, index=True) # Removed ForeignKey for flexibility, or keep if you have strict relations
    
    # Metadata for quick querying (Analytics)
    testcase_id = Column(String, index=True)
    title = Column(String)
    risk = Column(String)
    
    # 🟢 NEW: Store the WHOLE test case object here
    # If using Postgres: Use Column(JSON)
    # If using SQLite: Use Column(JSON) (modern SQLAlchemy handles it) or Column(Text)
    testcase_data = Column(JSON) 

    # Jira Details
    jira_key = Column(String)
    jira_url = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
