from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi import Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import shutil
from typing import Optional, List
from pydantic import BaseModel
import sys
from pathlib import Path

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

# --- Path setup ---
# Get the project's base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / 'static'
TEMPLATES_DIR = BASE_DIR / 'templates'

# Check if the directories exist and create them if they don't (for local dev)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount document directory to allow file downloads
documents_dir = settings.documents_path
os.makedirs(documents_dir, exist_ok=True)
app.mount("/documents", StaticFiles(directory=documents_dir), name="documents")


# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Initialize QA Generator
qa_generator = QAGenerator(rag_pipeline.retrieval_engine, rag_pipeline.llm_manager)

# Xóa bỏ biến lưu session trong bộ nhớ
# question_sessions = {}

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

class ProviderRequest(BaseModel):
    provider: str

# Dependency to get current user
def get_current_user(request: Request):
    user_info = request.session.get('user')
    if not user_info:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Thêm provider đã chọn vào thông tin user để dễ truy cập
    user_info['provider'] = request.session.get('provider', settings.default_provider)
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

@app.get("/admin/token-usage", response_class=HTMLResponse)
async def token_usage_page(
    request: Request,
    user: dict = Depends(get_admin_user),
    f_user: Optional[str] = None,  # Tạm thời nhận là string để xử lý giá trị rỗng
    f_provider: Optional[str] = "openai"
):
    """Trang quản lý token usage với bộ lọc và gom nhóm."""
    from collections import defaultdict

    # Chuyển đổi f_user rỗng thành None để query DB
    user_id_filter = int(f_user) if f_user and f_user.isdigit() else None

    # Lấy dữ liệu gốc, chưa gom nhóm
    # Bỏ category khỏi bộ lọc
    raw_usage_data = db.get_token_usage_summary(
        user_id=user_id_filter,
        category=None,
        provider=f_provider
    )

    # Gom nhóm dữ liệu bằng Python
    grouped_data = defaultdict(lambda: {'total_tokens': 0, 'categories': []})

    for item in raw_usage_data:
        key = (item['username'], item['provider'])

        # Cộng dồn token
        grouped_data[key]['total_tokens'] += item['total_tokens']

        # Thêm category và token của nó vào danh sách
        category_map = {
            'QA Generation': {'name': 'Tạo câu hỏi luyện tập', 'class': 'bg-primary'},
            'QA Evaluation': {'name': 'Đánh giá luyện tập', 'class': 'bg-info text-dark'},
            'Chat': {'name': 'Trò chuyện', 'class': 'bg-success'}
        }
        default_category = {'name': item['category'], 'class': 'bg-secondary'}

        category_info = category_map.get(item['category'], default_category)

        grouped_data[key]['categories'].append({
            'name': category_info['name'],
            'class': category_info['class'],
            'tokens': item['total_tokens']
        })

    # Chuyển đổi định dạng để dễ dàng hiển thị trên template
    usage_data = []
    for (username, provider), details in grouped_data.items():
        # Sắp xếp các category trong mỗi group, ví dụ theo tên
        details['categories'].sort(key=lambda x: x['name'])

        usage_data.append({
            'username': username,
            'provider': provider,
            'total_tokens': details['total_tokens'],
            'category_summary': details['categories'] # Bây giờ là list of dicts
        })

    # Sắp xếp lại kết quả
    usage_data.sort(key=lambda x: (x['username'], x['provider']))

    # Lấy dữ liệu cho các bộ lọc dropdown
    filter_data = db.get_distinct_token_usage_filters()

    return templates.TemplateResponse("token_usage.html", {
        "request": request,
        "usage_data": usage_data,
        "filters": filter_data,
        "current_filters": {
            "user": user_id_filter,
            "provider": f_provider
        },
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
async def history_page(request: Request, user: dict = Depends(get_current_user), page: int = 1, domain: str = 'all'):
    """Trang lịch sử học tập với phân trang và lọc theo lĩnh vực."""
    import math
    per_page = 10

    # Lấy danh sách các lĩnh vực mà người dùng đã làm
    user_domains = db.get_user_domains(user['id'])

    # Đếm và lấy lịch sử dựa trên lĩnh vực đã chọn
    total_records = db.count_user_qa_history(user['id'], domain=domain)
    history = db.get_user_qa_history(user['id'], page=page, per_page=per_page, domain=domain)

    total_pages = math.ceil(total_records / per_page) if total_records > 0 else 0

    return templates.TemplateResponse("history.html", {
        "request": request,
        "user": user,
        "history": history,
        "current_page": page,
        "total_pages": total_pages,
        "domains": user_domains,
        "current_domain": domain
    })

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
async def query(request: QueryRequest, user: dict = Depends(get_current_user)):
    """API endpoint để hỏi đáp (bây giờ là streaming)."""
    try:
        provider_to_use = user.get('provider')

        def stream_wrapper():
            # Ghi log token usage sẽ cần được xử lý riêng, vì chúng ta không có usage_data ở đây
            # Có thể ghi log sau khi stream kết thúc ở client, hoặc ước tính.
            # Hiện tại, tạm thời bỏ qua ghi log token cho streaming để đơn giản hóa.
            yield from rag_pipeline.stream_query(
                user_id=user['id'],
                question=request.question,
                use_fallback=request.use_fallback,
                provider=provider_to_use
            )

        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    except Exception as e:
        # StreamingResponse không thể raise HTTPException theo cách thông thường
        # Cần một cơ chế xử lý lỗi khác nếu cần
        async def error_stream():
            yield f"Error: {str(e)}"
        return StreamingResponse(error_stream(), media_type="text/event-stream", status_code=500)

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

@app.delete("/api/admin/clear-history", tags=["Admin"])
async def clear_all_history(user: dict = Depends(get_admin_user)):
    """Xóa toàn bộ lịch sử Q&A và các session đang tồn tại."""
    try:
        db.clear_qa_history()
        db.clear_qa_sessions()
        return {"message": "Đã xóa thành công toàn bộ lịch sử Q&A và các session."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa lịch sử: {str(e)}")

@app.get("/api/qa/generate")
async def generate_questions(user: dict = Depends(get_current_user), num_questions: int = 3, source_document: Optional[str] = None, provider: Optional[str] = None):
    """Tạo câu hỏi cho bài kiểm tra từ dữ liệu thực trong Pinecone"""
    try:
        if num_questions < 1 or num_questions > 10:
            raise HTTPException(status_code=400, detail="Số câu hỏi phải từ 1 đến 10")

        # Ưu tiên provider từ request, sau đó là session
        provider_to_use = provider or user.get('provider')

        # Tạo câu hỏi từ dữ liệu thực, có thể lọc theo tài liệu nguồn
        questions = qa_generator.generate_questions(
            user_id=user['id'],
            num_questions=num_questions,
            source_document=source_document if source_document and source_document != 'all' else None,
            provider=provider_to_use
        )

        # Tạo session ID để lưu trữ câu hỏi
        import uuid
        session_id = str(uuid.uuid4())

        # Lưu câu hỏi đầy đủ vào database thay vì bộ nhớ
        db.save_qa_session(session_id, questions)

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
        # Lấy câu hỏi từ database
        questions = db.get_qa_session(request.session_id)
        if not questions:
            raise HTTPException(status_code=404, detail="Session không tồn tại hoặc đã hết hạn")

        if len(questions) != len(request.user_answers):
            raise HTTPException(status_code=400, detail="Số câu trả lời không khớp với số câu hỏi")

        # Lấy provider từ session để đảm bảo tính nhất quán
        provider_to_use = user.get('provider')

        # Đánh giá câu trả lời
        evaluation = qa_generator.evaluate_answers(
            user_id=user['id'],
            questions=questions,
            user_answers=request.user_answers,
            provider=provider_to_use
        )

        # Thêm thông tin về nguồn dữ liệu
        evaluation['data_info'] = {
            'total_questions': len(questions),
            'questions_from_real_data': len([q for q in questions if q['type'] in ['extracted_from_data', 'real_data']]),
            'questions_from_llm': len([q for q in questions if q['type'] == 'generated']),
            'questions_fallback': len([q for q in questions if q['type'] == 'fallback'])
        }

        # Lưu lịch sử
        # Xác định lĩnh vực và tiêu đề từ câu hỏi đầu tiên
        domain = None
        domain_title = None
        if questions and 'source_doc' in questions[0] and questions[0]['source_doc']:
            source_doc = questions[0]['source_doc']
            source_path = source_doc.get('source')
            if source_path:
                domain = os.path.basename(source_path)
                # Lấy title từ metadata, nếu không có thì dùng tên file
                domain_title = source_doc.get('metadata', {}).get('title', domain)

        # Lưu lịch sử với thông tin lĩnh vực và tiêu đề
        db.add_qa_history(user['id'], evaluation, domain=domain, domain_title=domain_title)

        # Xóa session sau khi đánh giá
        db.delete_qa_session(request.session_id)

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
    request.session.clear() # Xóa toàn bộ session để đảm bảo sạch sẽ
    return {"message": "Logout successful"}

@app.post("/api/set-provider")
async def set_provider(req: ProviderRequest, request: Request, user: dict = Depends(get_current_user)):
    """Lưu provider người dùng chọn vào session."""
    # Kiểm tra xem provider có hợp lệ không (lấy từ llm_manager)
    if req.provider not in rag_pipeline.llm_manager.get_available_providers():
        raise HTTPException(status_code=400, detail="Invalid provider specified.")

    request.session['provider'] = req.provider
    return {"message": f"Provider set to {req.provider}"}

@app.get("/api/documents")
async def list_documents(user: dict = Depends(get_current_user)):
    """Lấy danh sách các lĩnh vực (sources) có trong database."""
    try:
        db_info = rag_pipeline.get_database_info()
        # Lấy thông tin chi tiết từ local metadata để có cả title và source
        all_docs = rag_pipeline.retrieval_engine.get_all_documents()

        # Tạo một dictionary để nhóm theo source, ưu tiên lấy title đẹp
        source_info = {}
        import os
        for doc in all_docs:
            source_path = doc.get('source')
            if not source_path:
                continue

            source_filename = os.path.basename(source_path)

            # Ưu tiên lấy title từ metadata, nếu không có thì dùng tên file
            display_text = doc.get('metadata', {}).get('title', source_filename)

            # Chỉ lưu lại nếu chưa có hoặc title hiện tại không phải là tên file
            if source_filename not in source_info or source_info[source_filename] == source_filename:
                 source_info[source_filename] = display_text

        # Chuyển thành định dạng list object cho frontend
        documents_list = [
            {"value": filename, "text": title}
            for filename, title in sorted(source_info.items())
        ]

        return {"documents": documents_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
