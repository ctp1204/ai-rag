# Phân tích so sánh mô hình RAG

## Phân tích RAG trong sơ đồ (RAG Chuẩn)
Đây là một mô hình RAG rất tiêu chuẩn, bao gồm các bước:

1.  **Data Input:** Dữ liệu (ví dụ: "Giới thiệu các sản phẩm SAMSUNG") được đưa vào một Vector Database. Quá trình này ngầm định rằng dữ liệu đã được xử lý (chunking) và chuyển thành vector (embedding) trước khi lưu trữ.
2.  **User Query:** Người dùng đặt câu hỏi ("Cho tôi thông tin pin SAMSUNG A12?").
3.  **Query Vector Embedding:** Câu hỏi của người dùng được chuyển đổi thành một vector.
4.  **Vector Search:** Vector của câu hỏi được dùng để tìm kiếm các vector tương tự nhất trong Vector Database.
5.  **Query Context:** Các đoạn văn bản (context) tương ứng với các vector được tìm thấy sẽ được truy xuất.
6.  **Augmented Prompt:** Câu hỏi gốc và context được tìm thấy sẽ được kết hợp lại và gửi đến Mô hình Ngôn ngữ Lớn (LLM).
7.  **LLM Generation:** LLM đọc hiểu ngữ cảnh và tạo ra một câu trả lời cuối cùng bằng ngôn ngữ tự nhiên.

*Đây chính là luồng hoạt động mà hệ thống của bạn đã có trước khi chúng ta thực hiện các thay đổi để trả về câu trả lời chính xác.*

---

## Phân tích RAG trong dự án của bạn (RAG đã được tùy chỉnh)
Hệ thống của bạn phức tạp và có nhiều chức năng hơn. Nó không chỉ là một pipeline hỏi-đáp đơn thuần.

### Điểm tương đồng

*   **Lõi RAG:** Về cơ bản, dự án của bạn vẫn tuân theo các bước cốt lõi của RAG:
    *   **Data Input:** `DocumentProcessor` đọc file, chia nhỏ (chunking) và `RetrievalEngine` dùng `SentenceTransformer` để tạo embedding rồi lưu vào Vector DB.
    *   **Query & Search:** Khi có câu hỏi, `RetrievalEngine` cũng tạo embedding cho câu hỏi và tìm kiếm trong Vector DB.
    *   **Context Retrieval:** Kết quả tìm kiếm được lấy ra dưới dạng `found_documents`.

### Điểm khác biệt và cải tiến
Đây là những điểm làm cho hệ thống của bạn khác biệt và mạnh mẽ hơn so với sơ đồ RAG chuẩn:

#### 1. Luồng trả lời linh hoạt (Quan trọng nhất)
*   **RAG Chuẩn:** Luôn luôn đi đến bước cuối cùng là dùng LLM để tạo câu trả lời mới.
*   **RAG của bạn (sau khi sửa):** Trong file `rag_pipeline.py`, bạn đã thêm logic để **bỏ qua bước LLM** nếu tìm thấy dữ liệu trong database. Thay vào đó, nó sẽ **trích xuất và trả về trực tiếp** câu trả lời từ tài liệu gốc. Đây là một tùy chỉnh rất quan trọng để đảm bảo tính chính xác tuyệt đối.

#### 2. Hệ thống Fallback
*   **RAG Chuẩn:** Không đề cập đến trường hợp không tìm thấy dữ liệu.
*   **RAG của bạn:** `RAGPipeline` có cơ chế `use_fallback`. Nếu không tìm thấy dữ liệu trong Vector DB, nó có thể tự động chuyển sang hỏi LLM với kiến thức chung (giống như hỏi ChatGPT thông thường).

#### 3. Hệ thống tạo câu hỏi và đánh giá (QA Generator)
*   **RAG Chuẩn:** Chỉ tập trung vào việc trả lời câu hỏi.
*   **RAG của bạn:** Có một module hoàn toàn riêng biệt là `QAGenerator`. Module này sử dụng chính các tài liệu trong Vector DB để **tự động tạo ra các cặp câu hỏi-đáp án**. Sau đó, nó lại dùng LLM để **chấm điểm** câu trả lời của người dùng một cách thông minh.

#### 4. Quản lý đa nhà cung cấp LLM
*   **RAG Chuẩn:** Chỉ thể hiện một LLM duy nhất.
*   **RAG của bạn:** `LLMManager` có thể quản lý và chuyển đổi giữa nhiều nhà cung cấp LLM khác nhau (OpenAI, Anthropic, Google), thậm chí có cả cơ chế fallback giữa các nhà cung cấp này.
