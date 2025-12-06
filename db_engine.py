import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "sentiment_history.db"


def init_db():
    """Khởi tạo cơ sở dữ liệu SQLite."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def save_result(text, sentiment):
    """Lưu kết quả phân loại vào DB."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
        (text, sentiment, timestamp),
    )
    conn.commit()
    conn.close()


def load_history(limit=50):
    """Tải lịch sử phân loại từ DB."""
    conn = sqlite3.connect(DB_NAME)

    if limit is None:
        query = "SELECT * FROM sentiments ORDER BY id DESC"
    else:
        query = f"SELECT * FROM sentiments ORDER BY id DESC LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
