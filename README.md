# Hệ thống Hỏi-Đáp và Tạo Bài kiểm tra thông minh (Advanced RAG System)

Đây là một hệ thống RAG (Retrieval-Augmented Generation) nâng cao, cho phép bạn xây dựng một AI assistant thông minh có thể trả lời câu hỏi dựa trên dữ liệu tùy chỉnh, đồng thời tích hợp một hệ thống tạo và đánh giá bài kiểm tra tự động.

## 🌟 Tính năng chính

-   **✅ Trả lời chính xác từ tài liệu:** Ưu tiên trích xuất câu trả lời trực tiếp từ tài liệu gốc để đảm bảo độ chính xác tuyệt đối.
-   **🧠 Hệ thống kiểm tra tự động:** Tự động tạo các cặp câu hỏi và đáp án từ kho tài liệu của bạn để tạo bài kiểm tra kiến thức.
-   **💯 Chấm điểm ngữ nghĩa thông minh:** Sử dụng LLM để đánh giá câu trả lời của người dùng dựa trên ý nghĩa, thay vì so khớp văn bản cứng nhắc.
-   **🤖 Fallback linh hoạt:** Tự động chuyển sang các nhà cung cấp LLM (Google, OpenAI, Anthropic) khi không tìm thấy dữ liệu trong tài liệu.
-   **🔑 API an toàn:** Hỗ trợ xác thực bằng API key cho các kết nối server-to-server.
-   **💾 Hỗ trợ đa dạng Vector DB:** Tương thích với Pinecone, ChromaDB, và FAISS.
-   **⚙️ Cấu hình linh hoạt:** Dễ dàng tùy chỉnh mọi thứ qua file `.env`.
-   **🚀 Sẵn sàng Deploy:** Cấu hình sẵn cho việc triển khai lên Render.com.

## 🚀 Hướng dẫn cài đặt và sử dụng

### 1. Clone và cài đặt dependencies

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

Sao chép file cấu hình mẫu và điền các thông tin cần thiết.
```bash
cp .env.example .env
```

Mở file `.env` và điền các API key cũng như cấu hình của bạn:
```env
# --- Bắt buộc ---
# Chọn ít nhất một nhà cung cấp LLM
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# --- Bắt buộc cho Vector DB là Pinecone ---
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment

# --- Bắt buộc cho giao diện web ---
SECRET_KEY=run_this_in_terminal_to_generate: openssl rand -hex 32

# --- Tùy chọn ---
# Nhà cung cấp LLM và Vector DB mặc định
LLM_PROVIDER=google
VECTOR_DB_TYPE=pinecone
PINECONE_INDEX_NAME=my-rag-index

# API Key cho kết nối server-to-server
SERVER_API_KEY=your_strong_secret_key_for_server_auth
```

### 3. Chạy ứng dụng локально

```bash
python main.py
```
Truy cập `http://localhost:8000` để sử dụng giao diện web.

## 🚀 Hướng dẫn Deploy lên Render.com

Dự án này đã được cấu hình sẵn để deploy dễ dàng thông qua "Infrastructure as Code".

1.  **Đẩy code lên GitHub/GitLab:** Đảm bảo bạn đã commit các file `render.yaml`, `build.sh`, và `requirements.txt`.
2.  **Tạo Blueprint trên Render:**
    *   Trên Dashboard Render, chọn **New +** > **Blueprint**.
    *   Kết nối với repository của bạn. Render sẽ tự động phát hiện và sử dụng file `render.yaml`.
3.  **Cung cấp Secrets:**
    *   Sau khi tạo resource, Render sẽ yêu cầu bạn nhập các giá trị cho các biến môi trường bí mật (ví dụ: `GOOGLE_API_KEY`, `PINECONE_API_KEY`...).
4.  **Deploy:**
    *   Lưu lại các secrets, Render sẽ tự động build và deploy ứng dụng của bạn.

## 🏗️ Kiến trúc hệ thống

Dự án được cấu trúc rõ ràng với các thành phần chính:

-   `src/api/main.py`: Lõi ứng dụng FastAPI, xử lý các request.
-   `src/core/rag_pipeline.py`: Điều phối luồng hỏi-đáp, quyết định khi nào truy xuất từ tài liệu, khi nào dùng fallback.
-   `src/core/qa_generator.py`: Chịu trách nhiệm tạo câu hỏi và chấm điểm thông minh.
-   `src/core/retrieval_engine.py`: Giao tiếp với Vector DB để tìm kiếm và truy xuất dữ liệu.
-   `src/core/llm_client.py`: Quản lý các kết nối đến những nhà cung cấp LLM khác nhau.
-   `templates/`: Chứa các file HTML cho giao diện người dùng.

Để xem sơ đồ kiến trúc trực quan, hãy tham khảo file `project_architecture.md`.

## 📝 API Endpoints

Hệ thống cung cấp 2 nhóm endpoint:

**1. Dành cho giao diện web (Xác thực bằng session):**
-   `GET /`: Trang chủ.
-   `POST /api/web/query`: Gửi câu hỏi từ giao diện web.
-   Các endpoint khác cho đăng nhập, đăng ký, quản lý...

**2. Dành cho Server-to-Server (Xác thực bằng `x-api-key`):**
-   `POST /api/query`: Gửi câu hỏi (yêu cầu `SERVER_API_KEY`).
-   `POST /api/upload-file`: Tải lên tài liệu (yêu cầu `SERVER_API_KEY`).

## 🤝 Đóng góp

1.  Fork the repository.
2.  Tạo một feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit các thay đổi của bạn (`git commit -m 'Add some AmazingFeature'`).
4.  Push lên branch (`git push origin feature/AmazingFeature`).
5.  Mở một Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
