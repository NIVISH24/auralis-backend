from uvicorn import run


if __name__ == "__main__":
    run("api:app", reload=True, reload_excludes=["uploads", "__pycache__"])
