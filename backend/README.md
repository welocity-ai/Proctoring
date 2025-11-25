# Backend - AI Proctoring API

This is the backend service for the AI Proctoring application, built with FastAPI.

## Project Structure

```
backend/
├── src/                    # Source code package
│   ├── __init__.py
│   ├── app.py             # FastAPI application
│   ├── config.py          # Configuration settings
│   ├── api/               # API routes
│   │   ├── __init__.py
│   │   └── routes.py      # Route handlers
│   ├── services/          # Business logic
│   │   ├── __init__.py
│   │   ├── session_service.py  # Session management
│   │   └── report_service.py   # Report generation
│   ├── models/            # Data models
│   │   ├── __init__.py
│   │   └── session.py     # Session model
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── face_utils.py  # Face detection utilities
│       └── voice_utils.py # Voice diarization utilities
├── tests/                 # Test files
│   ├── __init__.py
│   └── test_face_detection.py
├── reports/               # Generated PDF reports
├── screenshots/           # Captured screenshots
├── main.py               # Application entry point
├── app.py                # Backward compatibility shim
├── face_utils.py         # Backward compatibility shim
├── voice_utils.py        # Backward compatibility shim
├── requirements.txt      # Python dependencies
└── pyproject.toml        # Project configuration
```

## Running the Application

### Using main.py (Recommended)
```bash
python main.py
```

### Using uvicorn directly
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Using the compatibility shim
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `WebSocket /ws/{session_id}` - WebSocket connection for proctoring events
- `POST /generate_report/{session_id}` - Generate PDF report for a session

## Development

### Installation
```bash
pip install -r requirements.txt
```

### Project Structure Explanation

- **src/api/**: Contains all API route handlers
- **src/services/**: Contains business logic (session management, report generation)
- **src/models/**: Contains data models and schemas
- **src/utils/**: Contains utility functions (face detection, voice diarization)
- **src/config.py**: Contains configuration settings

### Backward Compatibility

The old `app.py`, `face_utils.py`, and `voice_utils.py` files in the root directory are maintained as compatibility shims that import from the new structured locations. For new code, import directly from the `src` package.

## Notes

- The `Procturing_analysis/` folder is maintained separately and is not part of this backend structure.
- Session data is stored in-memory. For production, consider using a persistent storage solution.
- Reports are generated as PDF files and stored in the `reports/` directory.


