@run:
    uv run -m pyrag.main
@urun:
    uv run uvicorn pyrag.main:app --reload
@lint:
    uv run ruff check --fix
    uv run ruff format