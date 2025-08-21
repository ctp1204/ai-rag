import google.generativeai as genai
from .llm_client import LLMClient
from config import settings

class GoogleClient(LLMClient):
    """Google Gemini API client"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.google_api_key
        self.model_name = model or "gemini-2.5-flash-preview-05-20"

        if not self.api_key:
            raise ValueError("Google API key is required")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> tuple[str, dict]:
        """Tạo response từ Google Gemini và trả về cả usage."""
        full_prompt = self._create_full_prompt(prompt, context)

        try:
            # Đếm input tokens
            input_tokens = self.model.count_tokens(full_prompt).total_tokens

            response = self.model.generate_content(full_prompt)
            response_text = response.text

            # Đếm output tokens (ước tính)
            output_tokens = self.model.count_tokens(response_text).total_tokens

            usage_dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "model_name": self.model_name,
                "provider": "google"
            }
            return response_text, usage_dict
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
