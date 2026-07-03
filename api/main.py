from fastapi import FastAPI
from datetime import datetime, timezone

app = FastAPI()

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "skincancer-api",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }