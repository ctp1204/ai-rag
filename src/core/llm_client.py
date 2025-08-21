from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import openai
import anthropic
from config import settings

class LLMClient(ABC):
    """Abstract base class cho LLM clients"""

    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> tuple[str, dict]:
        pass

    @abstractmethod
    def generate_streaming_response(self, prompt: str, context: str = "", **kwargs):
        pass

class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or "gpt-3.5-turbo"

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> tuple[str, dict]:
        """Tạo response từ OpenAI và trả về cả usage."""
        system_message = self._create_system_message(context)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000)
            )
            usage = response.usage
            usage_dict = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "model_name": self.model,
                "provider": "openai"
            }
            return response.choices[0].message.content, usage_dict
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    def generate_streaming_response(self, prompt: str, context: str = "", **kwargs):
        """Tạo response stream từ OpenAI."""
        system_message = self._create_system_message(context)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                yield content
        except Exception as e:
            raise Exception(f"OpenAI API streaming error: {str(e)}")

    def _create_system_message(self, context: str) -> str:
        """Tạo system message với context"""
        if context:
            return f"""Bạn là một AI assistant thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.

THÔNG TIN LIÊN QUAN:
{context}

Hướng dẫn:
- Trả lời bằng tiếng Việt
- Dựa chủ yếu vào thông tin được cung cấp
- Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó
- Trả lời một cách tự nhiên và dễ hiểu"""
        else:
            return """Bạn là một AI assistant thông minh. Hãy trả lời câu hỏi bằng tiếng Việt một cách tự nhiên và hữu ích."""

class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or "claude-3-sonnet-20240229"

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> tuple[str, dict]:
        """Tạo response từ Anthropic Claude và trả về cả usage."""
        system_message = self._create_system_message(context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            usage = response.usage
            usage_dict = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens,
                "model_name": self.model,
                "provider": "anthropic"
            }
            return response.content[0].text, usage_dict
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    def generate_streaming_response(self, prompt: str, context: str = "", **kwargs):
        """Tạo response stream từ Anthropic Claude."""
        system_message = self._create_system_message(context)
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise Exception(f"Anthropic API streaming error: {str(e)}")

    def _create_system_message(self, context: str) -> str:
        """Tạo system message với context"""
        if context:
            return f"""Bạn là một AI assistant thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.

THÔNG TIN LIÊN QUAN:
{context}

Hướng dẫn:
- Trả lời bằng tiếng Việt
- Dựa chủ yếu vào thông tin được cung cấp
- Nếu thông tin không đủ để trả lời, hãy nói rõ điều đó
- Trả lời một cách tự nhiên và dễ hiểu"""
        else:
            return """Bạn là một AI assistant thông minh. Hãy trả lời câu hỏi bằng tiếng Việt một cách tự nhiên và hữu ích."""

def create_llm_client(provider: str = None) -> LLMClient:
    """Factory function để tạo LLM client"""
    provider = provider or settings.llm_provider

    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "anthropic":
        return AnthropicClient()
    elif provider.lower() == "google":
        from .google_client import GoogleClient
        return GoogleClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

class LLMManager:
    """Manager để quản lý và tạo các LLM client."""

    def __init__(self):
        """Khởi tạo LLMManager."""
        # Kiểm tra xem có ít nhất một API key được cấu hình không
        if not self.get_available_providers():
            raise ValueError("No LLM API keys configured. Please set at least one in the .env file.")

    def generate_response(self, prompt: str, context: str = "", provider: Optional[str] = None, **kwargs) -> tuple[str, dict]:
        """
        Tạo response từ một provider cụ thể.
        Nếu provider không được chỉ định, sử dụng provider mặc định từ settings.
        """
        provider_to_use = provider or settings.default_provider

        try:
            client = create_llm_client(provider_to_use)
            return client.generate_response(prompt, context, **kwargs)
        except Exception as e:
            # Gói lại lỗi để cung cấp thêm ngữ cảnh
            raise Exception(f"Error with LLM provider '{provider_to_use}': {str(e)}") from e

    def generate_streaming_response(self, prompt: str, context: str = "", provider: Optional[str] = None, **kwargs):
        """Tạo response stream từ một provider cụ thể."""
        provider_to_use = provider or settings.default_provider
        try:
            client = create_llm_client(provider_to_use)
            return client.generate_streaming_response(prompt, context, **kwargs)
        except Exception as e:
            raise Exception(f"Error with LLM provider '{provider_to_use}': {str(e)}") from e

    def is_available(self) -> bool:
        """Kiểm tra xem có bất kỳ provider nào có sẵn không."""
        return bool(self.get_available_providers())

    def get_available_providers(self) -> list[str]:
        """Lấy danh sách các provider có sẵn dựa trên API keys."""
        providers = []
        if settings.openai_api_key:
            providers.append("openai")
        if settings.google_api_key:
            providers.append("google")
        if settings.anthropic_api_key:
            providers.append("anthropic")

        # Đảm bảo không có provider nào bị trùng
        return list(set(providers))
