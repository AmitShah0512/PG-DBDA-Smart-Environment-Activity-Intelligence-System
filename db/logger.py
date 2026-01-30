import sqlite3
from utils.config import DB_PATH

class ActionLogger:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS actions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            action TEXT,
            confidence REAL
        )
        """)
        self.conn.commit()

    def log(self, action, confidence):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO actions(action, confidence) VALUES (?,?)",
                    (action, float(confidence)))
        self.conn.commit()
        print(f"Logged: {action} ({confidence:.2f})")

    def close(self):
        self.conn.close()
