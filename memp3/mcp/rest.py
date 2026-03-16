"""Optional REST API for memp3 (used by `memp3 serve`)."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memp3.core.validators import ValidationError

logger = logging.getLogger(__name__)

app = FastAPI(title="memp3 REST API", description="AI memory encoded in sound")


def _get_storage():
    from memp3.core.storage import StorageManager
    return StorageManager()


class MemoryContent(BaseModel):
    content: str
    tags: str | None = None


class MemoryResponse(BaseModel):
    id: str
    content: str
    created_at: str


class SearchResponse(BaseModel):
    results: list[MemoryResponse]


@app.get("/")
async def root():
    return {"message": "memp3 REST API running", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/memories", response_model=MemoryResponse)
async def create_memory(memory: MemoryContent):
    try:
        storage = _get_storage()
        mem_id = storage.store(memory.content, memory.tags)
        info = storage.get_info(mem_id)
        return MemoryResponse(id=mem_id, content=memory.content, created_at=info["created_at"])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error creating memory")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/memories/{mem_id}", response_model=MemoryResponse)
async def get_memory(mem_id: str):
    try:
        storage = _get_storage()
        content = storage.retrieve(mem_id)
        info = storage.get_info(mem_id)
        return MemoryResponse(id=mem_id, content=content, created_at=info["created_at"])
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception:
        logger.exception("Error retrieving memory")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/memories", response_model=SearchResponse)
async def search_memories(query: str = ""):
    try:
        storage = _get_storage()
        results = storage.search(query) if query else storage.list_all()
        return SearchResponse(
            results=[
                MemoryResponse(id=r["id"], content=r["content"], created_at=r["created_at"])
                for r in results
            ]
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Error searching memories")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    try:
        storage = _get_storage()
        storage.delete(mem_id)
        return {"status": "deleted", "id": mem_id}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception:
        logger.exception("Error deleting memory")
        raise HTTPException(status_code=500, detail="Internal server error")
