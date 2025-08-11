import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

def check_pinecone_connection():
    """
    Checks the connection to Pinecone and verifies the index exists.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        print("Lỗi: Vui lòng đảm bảo các biến môi trường PINECONE_API_KEY và PINECONE_INDEX_NAME đã được thiết lập trong file .env.")
        return

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)

        # Check if the index exists
        if index_name in pc.list_indexes().names():
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print("✅ Kết nối tới Pinecone thành công!")
            print(f"✅ Index '{index_name}' đã được tìm thấy.")
            print("\n--- Thống kê Index ---")
            print(f"Số lượng vector: {stats.total_vector_count}")
            print(f"Số chiều: {stats.dimension}")
            print(f"Trạng thái Index: Sẵn sàng")
            print("---------------------\n")
        else:
            print(f"Lỗi: Index '{index_name}' không tồn tại trong tài khoản Pinecone của bạn.")
            print("Vui lòng kiểm tra lại tên index trong file .env hoặc tạo index mới trên Pinecone.")

    except Exception as e:
        print(f"❌ Kết nối tới Pinecone thất bại.")
        print(f"Lỗi chi tiết: {e}")
        print("\nVui lòng kiểm tra lại các thông tin sau:")
        print("1. PINECONE_API_KEY và PINECONE_ENVIRONMENT trong file .env có chính xác không.")
        print("2. Máy tính của bạn có kết nối Internet không.")

if __name__ == "__main__":
    check_pinecone_connection()
