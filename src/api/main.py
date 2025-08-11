from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
from typing import Optional, List
from pydantic import BaseModel
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.rag_pipeline import RAGPipeline
from config import settings

# Initialize FastAPI app
app = FastAPI(title="RAG System", description="Retrieval-Augmented Generation System")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    use_fallback: bool = True
    provider: Optional[str] = None

class TextDocumentRequest(BaseModel):
    text: str
    source: str = "web_input"
    title: Optional[str] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Trang admin để quản lý documents"""
    db_info = rag_pipeline.get_database_info()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "db_info": db_info
    })

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
