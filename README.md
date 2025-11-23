## Getting started
1. Copy Config
```bash
cp ./infra/example.config.coml ./infra/config.toml
```
2. Set up enviroments
```env
COMPOSE_FILE=./infra/docker-compose.yaml
CONFIG_PATH=./infra/config.toml
```
3. Run the qdrant, ollama via docker compose (if you need, you can also use cloud solutions)
```bash
docker compose up qdrant ollama -d
```
4. Start application via uvicorn (or via another ASGI server)
```bash
uv run uvicorn pyrag.main:app --reload
```

### TODO list
- Replace sync clients to async