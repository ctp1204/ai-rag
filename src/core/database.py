import sqlite3
import json
from datetime import datetime

DATABASE_NAME = "app_data.db"

def get_db_connection():
    """Tạo kết nối đến database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Khởi tạo các bảng trong database nếu chúng chưa tồn tại."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Bảng người dùng
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    # Bảng lịch sử Q&A
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        score INTEGER NOT NULL,
        total INTEGER NOT NULL,
        results TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# User functions
def add_user(username, password):
    """Thêm người dùng mới."""
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError: # Username đã tồn tại
        return False
    finally:
        conn.close()

def get_user(username):
    """Lấy thông tin người dùng bằng username."""
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return user

# History functions
def add_qa_history(user_id, evaluation):
    """Thêm một bản ghi lịch sử Q&A."""
    conn = get_db_connection()
    timestamp = datetime.now()
    results_json = json.dumps(evaluation['results'])

    conn.execute("""
    INSERT INTO qa_history (user_id, score, total, results, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, (user_id, evaluation['total_score'], evaluation['max_score'], results_json, timestamp))

    conn.commit()
    conn.close()

def get_user_qa_history(user_id):
    """Lấy toàn bộ lịch sử Q&A của người dùng."""
    conn = get_db_connection()
    history_rows = conn.execute("""
    SELECT * FROM qa_history
    WHERE user_id = ?
    ORDER BY timestamp DESC
    """, (user_id,)).fetchall()
    conn.close()

    # Chuyển đổi dữ liệu trả về
    history = []
    for row in history_rows:
        history.append({
            'total_score': row['score'],
            'max_score': row['total'],
            'results': json.loads(row['results']),
            'timestamp': row['timestamp']
        })
    return history

if __name__ == '__main__':
    init_db()
