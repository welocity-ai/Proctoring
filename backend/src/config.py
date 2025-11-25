"""Configuration settings for the backend application."""

import os
from typing import List

# CORS settings
CORS_ORIGINS: List[str] = ["*"]  # In production, replace with specific origins
CORS_METHODS: List[str] = ["*"]
CORS_HEADERS: List[str] = ["*"]

# Directories
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# API settings
API_TITLE = "AI Proctoring API"
API_VERSION = "1.0.0"

