# RAG System - Retrieval-Augmented Generation

Hệ thống RAG (Retrieval-Augmented Generation) cho phép bạn tạo một AI assistant thông minh có thể trả lời câu hỏi dựa trên dữ liệu tùy chỉnh của bạn, với khả năng fallback sang LLM API khi không tìm thấy dữ liệu liên quan.

## 🌟 Tính năng chính

- **📚 Xử lý đa định dạng**: Hỗ trợ PDF, DOCX, TXT
- **🔍 Tìm kiếm semantic**: Sử dụng vector embeddings để tìm kiếm thông tin liên quan
- **🤖 Fallback thông minh**: Tự động chuyển sang OpenAI/Anthropic API khi không có dữ liệu local
- **💾 Vector Database**: Hỗ trợ ChromaDB và FAISS
- **🌐 Web Interface**: Giao diện web đơn giản để upload dữ liệu và chat
- **⚙️ Cấu hình linh hoạt**: Dễ dàng tùy chỉnh qua file .env

## 🚀 Cài đặt nhanh

### 1. Clone và cài đặt dependencies

```bash
git clone <repository-url>
cd ai-retrieval-augmented-generation
pip install -r requirements.txt
```

### 2. Cấu hình API keys

```bash
cp .env.example .env
```

Chỉnh sửa file `.env` và thêm API key của bạn:

```env
# Chọn một trong các API sau
OPENAI_API_KEY=your_openai_api_key_here
# hoặc
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Cấu hình khác (tùy chọn)
VECTOR_DB_TYPE=chromadb
LLM_PROVIDER=openai
```

### 3. Chạy ứng dụng

```bash
python main.py
```

Truy cập: http://localhost:8000

## 📖 Hướng dẫn sử dụng

### Thêm dữ liệu

1. **Upload file**: Kéo thả hoặc chọn file PDF/DOCX/TXT
2. **Thêm text**: Nhập text trực tiếp vào hệ thống

### Chat với AI

1. Nhập câu hỏi vào ô chat
2. Hệ thống sẽ:
   - Tìm kiếm thông tin liên quan trong dữ liệu của bạn
   - Nếu tìm thấy: Trả lời dựa trên dữ liệu local
   - Nếu không tìm thấy: Sử dụng LLM API (nếu bật fallback)

### Quản lý dữ liệu

- Truy cập `/admin` để xem thống kê và quản lý documents
- Tìm kiếm, xóa documents cụ thể
- Xóa toàn bộ database nếu cần

## 🔧 Cấu hình nâng cao

### Vector Database

```env
# ChromaDB (khuyến nghị)
VECTOR_DB_TYPE=chromadb

# FAISS (nhanh hơn cho dataset lớn)
VECTOR_DB_TYPE=faiss
```

### LLM Provider

```env
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-3.5-turbo

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
```

### Retrieval Settings

```env
SIMILARITY_THRESHOLD=0.7  # Ngưỡng độ tương tự (0-1)
MAX_RETRIEVED_DOCS=5      # Số documents tối đa truy xuất
```

## 🏗️ Kiến trúc hệ thống

```
src/
├── core/
│   ├── document_processor.py  # Xử lý và embedding documents
│   ├── vector_database.py     # Vector database abstraction
│   ├── retrieval_engine.py    # Engine tìm kiếm semantic
│   ├── llm_client.py          # LLM API clients
│   └── rag_pipeline.py        # Main RAG pipeline
├── api/
│   └── main.py               # FastAPI application
└── web/
    └── ...                   # Web interface components
```

## 🧪 Testing

```bash
# Chạy tests cơ bản
python -m pytest tests/ -v

# Test từng component
python -c "from src.core.rag_pipeline import RAGPipeline; rag = RAGPipeline(); print(rag.health_check())"
```

## 📝 API Endpoints

- `GET /` - Trang chủ chat
- `GET /admin` - Trang quản lý
- `POST /api/query` - Chat với AI
- `POST /api/upload-file` - Upload file
- `POST /api/add-text` - Thêm text
- `GET /api/database-info` - Thông tin database
- `GET /api/search` - Tìm kiếm documents
- `DELETE /api/document/{id}` - Xóa document
- `DELETE /api/database` - Xóa toàn bộ database

## 🔍 Troubleshooting

### Lỗi thường gặp

1. **"No LLM clients available"**
   - Kiểm tra API key trong file `.env`
   - Đảm bảo đã cài đặt đúng dependencies

2. **"Unsupported file type"**
   - Chỉ hỗ trợ PDF, DOCX, TXT
   - Kiểm tra định dạng file

3. **Vector database errors**
   - Xóa thư mục `data/vector_db` và khởi động lại
   - Thử chuyển sang vector database khác

### Performance tuning

- Với dataset lớn (>10k documents): Sử dụng FAISS
- Với dataset nhỏ: ChromaDB đơn giản hơn
- Điều chỉnh `chunk_size` trong `document_processor.py`

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🆘 Hỗ trợ

Nếu gặp vấn đề, hãy tạo issue trên GitHub hoặc liên hệ qua email.
