from logging import INFO
from uvicorn import run


if __name__ == "__main__":
    run(
        "api:app",
        host="0.0.0.0",
        port=7000,
        log_level=INFO,
        reload=False,
        reload_excludes=["uploads", "__pycache__"],
    )
