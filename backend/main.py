"""Entry point for running the FastAPI backend server."""

import uvicorn
from src.app import app

def main():
    """Run the FastAPI application."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
