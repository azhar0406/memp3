import os
import sqlite3
import uuid
from datetime import datetime

class StorageManager:
    """Simple storage manager for memp3"""
    
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.expanduser("~/memp3")
        self.base_path = base_path
        self.memory_path = os.path.join(base_path, "memory")
        self.db_path = os.path.join(base_path, "index.db")
        
        # Create directories
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Initialize database
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def store(self, content, tags=None):
        """Store content as audio memory"""
        import soundfile as sf
        from memp3.core.encoder import SimpleEncoder
        
        # Generate ID and filename
        mem_id = str(uuid.uuid4())
        filename = f"{mem_id}.flac"
        filepath = os.path.join(self.memory_path, filename)
        
        # Encode content
        encoder = SimpleEncoder()
        signal = encoder.encode(content)
        
        # Save audio file
        sf.write(filepath, signal, encoder.sample_rate)
        
        # Save metadata
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (id, filename, content, tags)
            VALUES (?, ?, ?, ?)
        ''', (mem_id, filename, content, tags))
        conn.commit()
        conn.close()
        
        return mem_id
    
    def retrieve(self, mem_id):
        """Retrieve content by ID"""
        import soundfile as sf
        from memp3.core.encoder import SimpleEncoder
        
        # Get metadata
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT filename FROM memories WHERE id = ?', (mem_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise KeyError(f"Memory {mem_id} not found")
        
        filename = row[0]
        filepath = os.path.join(self.memory_path, filename)
        
        # Load and decode audio
        signal, sample_rate = sf.read(filepath)
        encoder = SimpleEncoder(sample_rate)
        content = encoder.decode(signal)
        
        return content
    
    def search(self, query):
        """Search memories by content"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at FROM memories 
            WHERE content LIKE ? ORDER BY created_at DESC
        ''', (f"%{query}%",))
        results = cursor.fetchall()
        conn.close()
        
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in results]
    
    def list_all(self):
        """List all memories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, content, created_at FROM memories ORDER BY created_at DESC')
        results = cursor.fetchall()
        conn.close()
        
        return [{"id": r[0], "content": r[1], "created_at": r[2]} for r in results]