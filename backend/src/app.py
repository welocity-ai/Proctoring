"""Main FastAPI application."""

import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from .config import CORS_ORIGINS, CORS_METHODS, CORS_HEADERS, API_TITLE, API_VERSION
from .api import routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title=API_TITLE, version=API_VERSION)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=CORS_METHODS,
    allow_headers=CORS_HEADERS,
)

# Register WebSocket route
@app.websocket("/ws/{session_id}")
async def websocket_route(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for proctoring events."""
    await routes.websocket_endpoint(websocket, session_id)

# Register report generation route
@app.post("/generate_report/{session_id}")
async def generate_report_route(session_id: str):
    """Generate PDF report for a session."""
    return await routes.generate_report_endpoint(session_id)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Proctoring API", "version": API_VERSION}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

