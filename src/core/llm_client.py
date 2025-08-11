from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import openai
import anthropic
from config import settings

class LLMClient(ABC):
    """Abstract base class cho LLM clients"""

    @abstractmethod
    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        pass

class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """Tạo response từ OpenAI"""
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
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

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
        self.model = model or settings.llm_model

        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> str:
        """Tạo response từ Anthropic Claude"""
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
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

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
    """Manager để quản lý LLM client với fallback logic"""

    def __init__(self, primary_client: Optional[LLMClient] = None):
        self.primary_client = primary_client
        self.fallback_clients = []

        # Tự động tạo clients nếu có API keys
        if not self.primary_client:
            try:
                self.primary_client = create_llm_client()
            except ValueError:
                pass

        # Thêm fallback clients
        self._setup_fallback_clients()

    def _setup_fallback_clients(self):
        """Thiết lập fallback clients"""
        # Thử tạo OpenAI client nếu không phải primary
        if settings.openai_api_key and (not self.primary_client or settings.llm_provider != "openai"):
            try:
                self.fallback_clients.append(OpenAIClient())
            except ValueError:
                pass

        # Thử tạo Anthropic client nếu không phải primary
        if settings.anthropic_api_key and (not self.primary_client or settings.llm_provider != "anthropic"):
            try:
                self.fallback_clients.append(AnthropicClient())
            except ValueError:
                pass

    def generate_response(self, prompt: str, context: str = "", provider: str = None, **kwargs) -> str:
        """Tạo response với fallback logic và lựa chọn provider"""

        # Nếu provider được chỉ định, chỉ dùng provider đó
        if provider:
            try:
                client = create_llm_client(provider)
                return client.generate_response(prompt, context, **kwargs)
            except Exception as e:
                raise Exception(f"Error with specified provider {provider}: {str(e)}")

        # Nếu không, dùng logic fallback mặc định
        clients_to_try = []
        if self.primary_client:
            clients_to_try.append(self.primary_client)
        clients_to_try.extend(self.fallback_clients)

        if not clients_to_try:
            raise Exception("No LLM clients available. Please configure API keys.")

        last_error = None
        for client in clients_to_try:
            try:
                return client.generate_response(prompt, context, **kwargs)
            except Exception as e:
                last_error = e
                continue

        raise Exception(f"All LLM clients failed. Last error: {str(last_error)}")

    def is_available(self) -> bool:
        """Kiểm tra xem có LLM client nào available không"""
        return self.primary_client is not None or len(self.fallback_clients) > 0
