import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from memp3.core.storage import StorageManager

app = FastAPI(title="memp3 MCP Server", description="AI memory encoded in sound")
storage = StorageManager()

class MemoryContent(BaseModel):
    content: str
    tags: Optional[str] = None

class MemoryResponse(BaseModel):
    id: str
    content: str
    created_at: str

class SearchResponse(BaseModel):
    results: List[MemoryResponse]

@app.get("/")
async def root():
    return {"message": "memp3 MCP server running", "version": "0.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/memories", response_model=MemoryResponse)
async def create_memory(memory: MemoryContent):
    """Create a new memory"""
    try:
        mem_id = storage.store(memory.content, memory.tags)
        # Retrieve the created memory to get the full details
        content = storage.retrieve(mem_id)
        # Get metadata from database
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT created_at FROM memories WHERE id = ?', (mem_id,))
        row = cursor.fetchone()
        conn.close()
        
        return MemoryResponse(
            id=mem_id,
            content=content,
            created_at=row[0] if row else ""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{mem_id}", response_model=MemoryResponse)
async def get_memory(mem_id: str):
    """Retrieve a memory by ID"""
    try:
        content = storage.retrieve(mem_id)
        # Get metadata from database
        import sqlite3
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT created_at FROM memories WHERE id = ?', (mem_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")
            
        return MemoryResponse(
            id=mem_id,
            content=content,
            created_at=row[0]
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", response_model=SearchResponse)
async def search_memories(query: str = ""):
    """Search memories"""
    try:
        if query:
            results = storage.search(query)
        else:
            results = storage.list_all()
            
        memories = []
        for result in results:
            memories.append(MemoryResponse(
                id=result['id'],
                content=result['content'],
                created_at=result['created_at']
            ))
            
        return SearchResponse(results=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    """Delete a memory (not implemented in MVP)"""
    raise HTTPException(status_code=501, detail="Not implemented")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3141)