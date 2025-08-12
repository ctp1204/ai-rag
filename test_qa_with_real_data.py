#!/usr/bin/env python3
"""
Test Q&A system với dữ liệu thực từ Pinecone
"""

import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.core.rag_pipeline import RAGPipeline
from src.core.qa_generator import QAGenerator

def main():
    print("🧪 Test Q&A System với Dữ liệu Thực")
    print("=" * 60)
    
    # Initialize components
    print("📊 Khởi tạo RAG pipeline...")
    rag = RAGPipeline()
    
    print("🎯 Khởi tạo QA Generator...")
    qa_generator = QAGenerator(rag.retrieval_engine, rag.llm_manager)
    
    # Check database
    db_info = rag.get_database_info()
    print(f"📚 Database: {db_info['vector_db_type']}")
    print(f"📄 Tổng documents: {db_info['total_documents']}")
    
    if db_info['total_documents'] == 0:
        print("❌ Database trống! Hãy upload dữ liệu trước:")
        print("   python3 upload_sample_data.py")
        return
    
    # Show data sources
    if db_info.get('sources'):
        print("📋 Nguồn dữ liệu:")
        for source, count in db_info['sources'].items():
            print(f"   - {source}: {count} chunks")
    
    # Get all documents to analyze
    print(f"\n🔍 Phân tích dữ liệu để tạo câu hỏi...")
    all_docs = rag.retrieval_engine.get_all_documents()
    
    print(f"📊 Chi tiết documents:")
    for i, doc in enumerate(all_docs[:5], 1):  # Show first 5 docs
        text_preview = doc.get('text', '')[:100] + "..." if len(doc.get('text', '')) > 100 else doc.get('text', '')
        print(f"   {i}. Source: {doc.get('source', 'Unknown')}")
        print(f"      Text: {text_preview}")
        print(f"      Chunk: {doc.get('chunk_index', 'N/A')}")
        print()
    
    if len(all_docs) > 5:
        print(f"   ... và {len(all_docs) - 5} documents khác")
    
    # Test question generation
    print(f"\n🎲 Tạo câu hỏi từ dữ liệu thực...")
    
    try:
        # Test với số lượng câu hỏi khác nhau
        for num_q in [3, 5]:
            print(f"\n--- Tạo {num_q} câu hỏi ---")
            questions = qa_generator.generate_questions(num_q)
            
            print(f"✅ Đã tạo {len(questions)} câu hỏi:")
            
            for i, q in enumerate(questions, 1):
                print(f"\n{i}. 📝 {q['question']}")
                print(f"   ✅ Đáp án: {q['correct_answer']}")
                print(f"   🏷️ Loại: {q['type']}")
                
                # Show source info if available
                if q.get('source_doc'):
                    source_doc = q['source_doc']
                    print(f"   📄 Nguồn: {source_doc.get('source', 'Unknown')} (chunk {source_doc.get('chunk_index', 'N/A')})")
                    
                    # Show a bit of source text
                    source_text = source_doc.get('text', '')
                    if source_text:
                        preview = source_text[:150] + "..." if len(source_text) > 150 else source_text
                        print(f"   📖 Text gốc: {preview}")
        
        # Test evaluation với câu trả lời mẫu
        print(f"\n🧪 Test đánh giá với câu trả lời mẫu...")
        
        # Lấy 3 câu hỏi để test
        test_questions = qa_generator.generate_questions(3)
        
        # Tạo câu trả lời test (một số đúng, một số sai)
        test_answers = []
        for q in test_questions:
            correct_answer = q['correct_answer']
            
            # Tạo câu trả lời với độ chính xác khác nhau
            if len(test_answers) == 0:
                # Câu 1: Trả lời chính xác
                test_answers.append(correct_answer)
            elif len(test_answers) == 1:
                # Câu 2: Trả lời gần đúng
                test_answers.append(correct_answer[:len(correct_answer)//2])
            else:
                # Câu 3: Trả lời sai
                test_answers.append("Không biết")
        
        print(f"\n📝 Câu trả lời test:")
        for i, (q, answer) in enumerate(zip(test_questions, test_answers), 1):
            print(f"   {i}. Q: {q['question']}")
            print(f"      A: {answer}")
            print(f"      Correct: {q['correct_answer']}")
            print()
        
        # Evaluate
        evaluation = qa_generator.evaluate_answers(test_questions, test_answers)
        
        print(f"📊 Kết quả đánh giá:")
        print(f"   🎯 Điểm: {evaluation['total_score']:.1f}/{evaluation['max_score']}")
        print(f"   📈 Tỷ lệ: {evaluation['percentage']}%")
        print(f"   💬 Nhận xét: {evaluation['overall_feedback']}")
        
        print(f"\n📋 Chi tiết từng câu:")
        for i, result in enumerate(evaluation['results'], 1):
            score_emoji = "🟢" if result['score'] >= 0.7 else "🟡" if result['score'] >= 0.5 else "🔴"
            print(f"   {i}. {score_emoji} {result['score']*100:.0f}% - {result['feedback']}")
        
        print(f"\n✅ Test hoàn thành!")
        print(f"\n💡 Để test trên web interface:")
        print("1. Chạy server: python3 -m uvicorn src.api.main:app --reload")
        print("2. Truy cập: http://localhost:8000/qa")
        print("3. Chọn số câu hỏi và bắt đầu kiểm tra")
        
        print(f"\n🎯 Lưu ý:")
        print("- Câu hỏi được tạo từ dữ liệu thực trong Pinecone")
        print("- Hệ thống phân tích text để tìm thông tin cụ thể")
        print("- Fallback sang câu hỏi tổng quát nếu không đủ dữ liệu")
        
    except Exception as e:
        print(f"❌ Lỗi test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
