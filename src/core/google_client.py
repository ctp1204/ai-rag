import google.generativeai as genai
from .llm_client import LLMClient
from config import settings

class GoogleClient(LLMClient):
    """Google Gemini API client"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.google_api_key
        self.model_name = model or settings.gemini_model

        if not self.api_key:
            raise ValueError("Google API key is required")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """Tạo response từ Google Gemini"""
        full_prompt = self._create_full_prompt(prompt, context)

        try:
            # Sử dụng streaming để có thể xử lý các response lớn
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Google Gemini API error: {str(e)}")

    def _create_full_prompt(self, prompt: str, context: str) -> str:
        """Tạo prompt đầy đủ với context"""
        if context:
            return f"""Dựa vào thông tin được cung cấp dưới đây, hãy trả lời câu hỏi một cách tự nhiên và dễ hiểu.

            THÔNG TIN LIÊN QUAN:
            ---
            {context}
            ---

            CÂU HỎI: {prompt}

            TRẢ LỜI:"""
        else:
            return prompt
