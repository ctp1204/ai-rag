from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi import Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import shutil
from typing import Optional, List
from pydantic import BaseModel
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.rag_pipeline import RAGPipeline
from src.core.qa_generator import QAGenerator
from src.core import database as db
from config import settings

# Initialize FastAPI app
app = FastAPI(title="RAG System", description="Retrieval-Augmented Generation System")
secret = os.getenv("SECRET_KEY", "default-secret-key")
app.add_middleware(SessionMiddleware, secret_key=secret)

# Initialize database
db.init_db()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Initialize QA Generator
qa_generator = QAGenerator(rag_pipeline.retrieval_engine, rag_pipeline.llm_manager)

# Store for temporary question sessions
question_sessions = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    use_fallback: bool = True
    provider: Optional[str] = None

class TextDocumentRequest(BaseModel):
    text: str
    source: str = "web_input"
    title: Optional[str] = None

class QAEvaluationRequest(BaseModel):
    session_id: str
    user_answers: List[str]

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

# Dependency to get current user
def get_current_user(request: Request):
    user_info = request.session.get('user')
    if not user_info:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_info

# Dependency for admin users
def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Not authorized")
    return current_user

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: dict = Depends(get_current_user)):
    """Trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/qa", response_class=HTMLResponse)
async def qa_page(request: Request, user: dict = Depends(get_current_user)):
    """Trang Q&A học tập"""
    return templates.TemplateResponse("qa.html", {"request": request, "user": user})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request, user: dict = Depends(get_admin_user)):
    """Trang admin để quản lý documents"""
    db_info = rag_pipeline.get_database_info()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "db_info": db_info,
        "user": user
    })

@app.get("/admin/users", response_class=HTMLResponse)
async def user_management(request: Request, user: dict = Depends(get_admin_user)):
    """Trang quản lý người dùng"""
    all_users = db.get_all_users()
    return templates.TemplateResponse("user_management.html", {"request": request, "users": all_users, "user": user})

@app.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def view_user_history(user_id: int, request: Request, user: dict = Depends(get_admin_user)):
    """Xem lịch sử của một người dùng cụ thể"""
    conn = db.get_db_connection()
    target_user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()

    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    history = db.get_user_qa_history(user_id)
    return templates.TemplateResponse("user_history.html", {
        "request": request,
        "history": history,
        "target_user": target_user,
        "user": user
    })

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Trang đăng nhập"""
    user = request.session.get('user')
    return templates.TemplateResponse("login.html", {"request": request, "user": user})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Trang đăng ký"""
    user = request.session.get('user')
    return templates.TemplateResponse("register.html", {"request": request, "user": user})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, user: dict = Depends(get_current_user)):
    """Trang lịch sử học tập"""
    history = db.get_user_qa_history(user['id'])
    return templates.TemplateResponse("history.html", {"request": request, "user": user, "history": history})

# Override the default 401 error handler to redirect to login
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        return RedirectResponse(url="/login")
    # You might want to handle other HTTPExceptions differently
    # For now, let's re-raise for other codes to use FastAPI's default
    raise exc

@app.post("/api/query")
async def query(request: QueryRequest):
    """API endpoint để hỏi đáp"""
    try:
        result = rag_pipeline.query(
            question=request.question,
            use_fallback=request.use_fallback,
            provider=request.provider
        )
        # Lấy nguồn đã được xác định từ pipeline và đưa lên cấp cao nhất
        result['source'] = result.get('metadata', {}).get('source', 'Không xác định')
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...), title: str = Form(None)):
    """Upload và xử lý file"""
    try:
        # Kiểm tra file type
        allowed_extensions = ['.txt', '.pdf', '.docx']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
            )

        # Lưu file tạm thời
        os.makedirs(settings.documents_path, exist_ok=True)
        file_path = os.path.join(settings.documents_path, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Xử lý file với RAG pipeline
        metadata = {"title": title or file.filename, "uploaded_via": "web"}
        result = rag_pipeline.add_document_from_file(file_path, metadata)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/add-text")
async def add_text(request: TextDocumentRequest):
    """Thêm text document trực tiếp"""
    try:
        metadata = {"title": request.title or "Text Input", "added_via": "web"}
        result = rag_pipeline.add_document_from_text(
            text=request.text,
            source=request.source,
            metadata=metadata
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database-info")
async def get_database_info():
    """Lấy thông tin database"""
    return rag_pipeline.get_database_info()

@app.get("/api/search")
async def search_documents(q: str, k: int = 5):
    """Tìm kiếm documents"""
    try:
        results = rag_pipeline.search_documents(q, k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/document/{doc_id}")
async def delete_document(doc_id: str):
    """Xóa document"""
    try:
        result = rag_pipeline.delete_document(doc_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/database")
async def clear_database():
    """Xóa toàn bộ database"""
    try:
        result = rag_pipeline.clear_database()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check"""
    return rag_pipeline.health_check()

@app.get("/api/qa/generate")
async def generate_questions(num_questions: int = 3):
    """Tạo câu hỏi cho bài kiểm tra từ dữ liệu thực trong Pinecone"""
    try:
        if num_questions < 1 or num_questions > 10:
            raise HTTPException(status_code=400, detail="Số câu hỏi phải từ 1 đến 10")

        # Tạo câu hỏi từ dữ liệu thực
        questions = qa_generator.generate_questions(num_questions)

        # Tạo session ID để lưu trữ câu hỏi
        import uuid
        session_id = str(uuid.uuid4())

        # Lưu câu hỏi đầy đủ vào session
        question_sessions[session_id] = questions

        # Chỉ trả về câu hỏi, không trả về đáp án
        response_questions = []
        for q in questions:
            response_questions.append({
                'id': q['id'],
                'question': q['question'],
                'type': q['type'],
                'source': q.get('source_doc', {}).get('source', 'Unknown') if q.get('source_doc') else 'System'
            })

        return {
            'session_id': session_id,
            'questions': response_questions,
            'total': len(response_questions),
            'data_source': 'pinecone_real_data'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/qa/evaluate")
async def evaluate_answers(request: QAEvaluationRequest, user: dict = Depends(get_current_user)):
    """Đánh giá câu trả lời của user dựa trên dữ liệu thực"""
    try:
        # Lấy câu hỏi từ session
        if request.session_id not in question_sessions:
            raise HTTPException(status_code=404, detail="Session không tồn tại hoặc đã hết hạn")

        questions = question_sessions[request.session_id]

        if len(questions) != len(request.user_answers):
            raise HTTPException(status_code=400, detail="Số câu trả lời không khớp với số câu hỏi")

        # Đánh giá câu trả lời
        evaluation = qa_generator.evaluate_answers(questions, request.user_answers)

        # Thêm thông tin về nguồn dữ liệu
        evaluation['data_info'] = {
            'total_questions': len(questions),
            'questions_from_real_data': len([q for q in questions if q['type'] in ['extracted_from_data', 'real_data']]),
            'questions_from_llm': len([q for q in questions if q['type'] == 'generated']),
            'questions_fallback': len([q for q in questions if q['type'] == 'fallback'])
        }

        # Lưu lịch sử
        db.add_qa_history(user['id'], evaluation)

        # Xóa session sau khi đánh giá (optional)
        # del question_sessions[request.session_id]

        return evaluation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login(request: Request, login_request: LoginRequest):
    """API endpoint để đăng nhập"""
    user_db = db.get_user(login_request.username)
    # In a real app, you'd use hashed passwords
    if not user_db or not user_db["password"] == login_request.password:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Store user info in session
    request.session['user'] = {
        "id": user_db['id'],
        "username": user_db['username'],
        "role": user_db['role']
    }
    return {"message": "Login successful"}

@app.post("/api/register")
async def register(register_request: RegisterRequest):
    """API endpoint để đăng ký"""
    # In a real app, hash the password
    success = db.add_user(register_request.username, register_request.password)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )
    return {"message": "User created successfully"}

@app.get("/api/logout")
async def logout(request: Request):
    """API endpoint để đăng xuất"""
    request.session.pop('user', None)
    return {"message": "Logout successful"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
