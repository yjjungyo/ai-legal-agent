import sqlite3, os
from typing import List, Tuple
from src.config import SQLITE_PATH
from src.utils.logging_utils import setup_logging
logger = setup_logging()

def get_conn():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS chats (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      chat_id INTEGER NOT NULL,
      role TEXT CHECK(role IN ('user','assistant')) NOT NULL,
      content TEXT NOT NULL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
    USING fts5(content, chat_id UNINDEXED, message_id UNINDEXED, tokenize = 'unicode61');
    CREATE TRIGGER IF NOT EXISTS trg_messages_ai AFTER INSERT ON messages
    BEGIN
      INSERT INTO messages_fts(rowid, content, chat_id, message_id)
      VALUES (new.id, new.content, new.chat_id, new.id);
    END;
    """)
    conn.commit()
    conn.close()

def new_chat(title: str = "새 채팅") -> int:
    conn = get_conn()
    cur = conn.execute("INSERT INTO chats(title) VALUES(?)", (title,))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    logger.info(f"[DB] 새 채팅 생성: id={cid}, title='{title}'")
    return cid

def rename_chat(chat_id: int, title: str):
    conn = get_conn()
    conn.execute("UPDATE chats SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()

def add_message(chat_id: int, role: str, content: str):
    conn = get_conn()
    conn.execute("INSERT INTO messages(chat_id, role, content) VALUES(?,?,?)", (chat_id, role, content))
    conn.execute("UPDATE chats SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()
    logger.debug(f"[DB] 메시지 추가: chat={chat_id}, role={role}, len={len(content)}")

def list_chats(limit: int = 50) -> List[Tuple[int, str, str]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, COALESCE(title,'(제목 없음)'), created_at FROM chats ORDER BY updated_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows

def get_chat_messages(chat_id: int) -> List[Tuple[str,str]]:
    conn = get_conn()
    rows = conn.execute("SELECT role, content FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)).fetchall()
    conn.close()
    return rows

def search_messages(q: str, limit: int = 50) -> List[Tuple[int,str]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT chat_id, snippet(messages_fts, 0, '[', ']', '...', 8) "
        "FROM messages_fts WHERE messages_fts MATCH ? LIMIT ?",
        (q, limit)
    ).fetchall()
    conn.close()
    return rows

