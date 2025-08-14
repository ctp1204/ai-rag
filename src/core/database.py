import sqlite3
import json
from datetime import datetime
from typing import Optional

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
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
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

    # Bảng lưu session câu hỏi
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS qa_sessions (
        id TEXT PRIMARY KEY,
        questions TEXT NOT NULL,
        created_at DATETIME NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")

# User functions
def add_user(username, password, role='user'):
    """Thêm người dùng mới."""
    conn = get_db_connection()
    try:
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, password, role))
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

def get_all_users():
    """Lấy danh sách tất cả người dùng."""
    conn = get_db_connection()
    users = conn.execute("SELECT id, username, role FROM users ORDER BY username").fetchall()
    conn.close()
    return users

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

def get_user_qa_history(user_id: int, page: int = 1, per_page: int = 10):
    """Lấy lịch sử Q&A của người dùng với phân trang."""
    conn = get_db_connection()
    offset = (page - 1) * per_page
    history_rows = conn.execute("""
        SELECT * FROM qa_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (user_id, per_page, offset)).fetchall()
    conn.close()

    # Chuyển đổi dữ liệu trả về
    history = []
    for row in history_rows:
        history.append({
            'id': row['id'],
            'total_score': row['score'],
            'max_score': row['total'],
            'results': json.loads(row['results']),
            'timestamp': row['timestamp']
        })
    return history

def count_user_qa_history(user_id: int) -> int:
    """Đếm tổng số bản ghi lịch sử của người dùng."""
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(id) FROM qa_history WHERE user_id = ?", (user_id,)).fetchone()[0]
    conn.close()
    return count

# --- QA Session Functions ---

def save_qa_session(session_id: str, questions: list):
    """Lưu một session câu hỏi vào database."""
    conn = get_db_connection()
    questions_json = json.dumps(questions)
    timestamp = datetime.now()
    conn.execute(
        "INSERT INTO qa_sessions (id, questions, created_at) VALUES (?, ?, ?)",
        (session_id, questions_json, timestamp)
    )
    conn.commit()
    conn.close()

def get_qa_session(session_id: str) -> Optional[list]:
    """Lấy một session câu hỏi từ database."""
    conn = get_db_connection()
    row = conn.execute("SELECT questions FROM qa_sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    if row:
        return json.loads(row['questions'])
    return None

def delete_qa_session(session_id: str):
    """Xóa một session câu hỏi khỏi database."""
    conn = get_db_connection()
    conn.execute("DELETE FROM qa_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    init_db()
