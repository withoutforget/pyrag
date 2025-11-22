@run:
    uv run -m pyrag.main
@urun:
    uv run uvicorn pyrag.main:app --reload