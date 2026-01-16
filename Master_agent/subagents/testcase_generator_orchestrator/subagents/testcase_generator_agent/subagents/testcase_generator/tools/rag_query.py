# rag_query_multi.py
"""
Multi-corpus RAG query tool for Vertex AI RAG Engine.
Drop-in compatible with your existing patterns; adds support for multiple corpora.
"""

import logging
from typing import List, Dict, Any

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

DEFAULT_DISTANCE_THRESHOLD = 0.8
DEFAULT_TOP_K = 5

from .utils import check_corpus_exists, get_corpus_resource_name

from vertexai import rag
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import os
from models import SessionLocal, Document

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
RAG_CORPUS_ID = os.getenv("DATA_STORE_ID")
RAG_CORPUS_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"


@retry(
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(ResourceExhausted)
)
def rag_query(
    corpora: List[str],  # display names; may be empty to use current_corpus
    query: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Query one or more Vertex AI RAG corpora and return aggregated results.

    Args:
      corpora: List of corpus display names. If empty, uses tool_context.state["current_corpus"].
      query: User query text
      tool_context: ADK ToolContext

    Returns:
      dict: status, message, corpora, results, results_count
    """
    try:
        # Resolve default from state
        if not corpora:
            current = tool_context.state.get("current_corpus")
            corpora = [current] if current else []

        if not corpora:
            return {
                "status": "error",
                "message": "No corpus specified and no current corpus set.",
                "query": query,
                "corpora": corpora,
                "results": [],
                "results_count": 0,
            }
            
            
        session_id = None
        user_id = None

        # Try multiple ways to access session info
        if tool_context:
            try:
                # Try accessing from invocation_context
                if hasattr(tool_context, '_invocation_context'):
                    inv_ctx = tool_context._invocation_context
                    if inv_ctx and hasattr(inv_ctx, 'session'):
                        session = inv_ctx.session
                        session_id = session.id if hasattr(session, 'id') else None
                        user_id = session.user_id if hasattr(session, 'user_id') else None

                # Fallback: try state
                if not user_id and hasattr(tool_context, 'state'):
                    state = tool_context.state
                    user_id = state.get('user_id') or state.get('user_name')

                # Another fallback: direct attributes
                if not session_id and hasattr(tool_context, 'session_id'):
                    session_id = tool_context.session_id
                if not user_id and hasattr(tool_context, 'user_id'):
                    user_id = tool_context.user_id

            except Exception as ctx_error:
                print(f"⚠ Error extracting context: {ctx_error}")
                logging.info(f"\n\n\nError extracting context: {ctx_error}\n\n\n")

        print(f"\n\n❌ Session context - session_id: {session_id}, user_id: {user_id}\n\n")
        
        logging.info(f"\n\nRAG Query Context - session_id: {session_id}, user_id: {user_id}\n\n")


        if not session_id or not user_id:
            print(f"❌ Missing context - session_id: {session_id}, user_id: {user_id}")
            return {
                "status": "error",
                "message": "Could not determine session or user context. Please ensure you're in an active session."
            }

        print(f"🔍 RAG Query - Session: {session_id}, User: {user_id}, Question: {query}")

        # Query PostgreSQL for active documents
        db = SessionLocal()
        try:
            active_docs = db.query(Document).filter(
                Document.session_id == session_id,
                Document.user_id == user_id,
                Document.is_active == True,
                Document.status == 'active',
                Document.rag_file_id.isnot(None)
            ).all()

            if not active_docs:
                print("⚠ No active documents found")
                return {
                    "status": "no_documents",
                    "message": "No active documents found in this session. Please upload and activate documents first."
                }

            rag_file_ids = []
            for doc in active_docs:
                # Extract just the file ID from the full resource name
                file_id = doc.rag_file_id.split('/')[-1] if '/' in doc.rag_file_id else doc.rag_file_id
                rag_file_ids.append(file_id)

            print(f"✅ Found {len(active_docs)} active documents:")
            for doc, file_id in zip(active_docs, rag_file_ids):
                print(f"\n\n   - {doc.filename} (File ID: {file_id})\n")
        except Exception as db_error:
            print(f"❌ Database query error: {db_error}")
            return {
                "status": "error",
                "message": f"Database error while retrieving active documents: {db_error}"
            }


        # Validate and resolve resource names
        valid_display_names: List[str] = []
        resources: List[rag.RagResource] = []
        invalid: List[str] = []

        for name in corpora:
            if not check_corpus_exists(name, tool_context):
                invalid.append(name)
                continue
            rn = get_corpus_resource_name(name)
            valid_display_names.append(name)
            if(name=="requirements"):
                resources.append(rag.RagResource(rag_corpus=rn, rag_file_ids=rag_file_ids))
            else:
                resources.append(rag.RagResource(rag_corpus=rn))
                
        print(f"\n\n✅ resources to query: {resources}\n\n")

        if not resources:
            return {
                "status": "error",
                "message": f"No valid corpora found. Invalid: {invalid}",
                "query": query,
                "corpora": corpora,
                "results": [],
                "results_count": 0,
            }
            
            
            
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )

        # Single retrieval across multiple corpora
        response = rag.retrieval_query(
            rag_resources=resources,
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        results: List[Dict[str, Any]] = []
        # Adapt to observed response shape in your current code
        # Prefer a robust parse that works if response.contexts is either a list or an object
        ctxs = getattr(response, "contexts", None)
        if ctxs:
            # Try iterable of context objects
            iterable = getattr(ctxs, "contexts", ctxs)
            for ctx in iterable:
                results.append({
                    "source_uri": getattr(ctx, "source_uri", "") or "",
                    "source_name": getattr(ctx, "source_display_name", "") or "",
                    "text": getattr(ctx, "text", "") or "",
                    "score": getattr(ctx, "score", 0.0) or 0.0,
                })

        if not results:
            return {
                "status": "warning",
                "message": f"No results found in corpora {valid_display_names} for query.",
                "query": query,
                "corpora": valid_display_names,
                "invalid_corpora": invalid,
                "results": [],
                "results_count": 0,
            }

        return {
            "status": "success",
            "message": f"Successfully queried corpora {valid_display_names}.",
            "query": query,
            "corpora": valid_display_names,
            "invalid_corpora": invalid,
            "results": results,
            "results_count": len(results),
        }

    except Exception as e:
        logging.error("Multi-corpus query error: %s", e)
        return {
            "status": "error",
            "message": f"Error querying corpora: {str(e)}",
            "query": query,
            "corpora": corpora,
            "results": [],
            "results_count": 0,
        }