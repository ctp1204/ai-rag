from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import openai
import anthropic
import google.generativeai as genai
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
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, prompt: str, context: str = "", **kwargs) -> tuple[str, dict]:
        system_message = self._create_system_message(context)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
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
        system_message = self._create_system_message(context)
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
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
        system_message = self._create_system_message(context)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=[{"role": "user", "content": prompt}]
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
        full_prompt = self._create_full_prompt(prompt, context)
        try:
            input_tokens = self.model.count_tokens(full_prompt).total_tokens
            response = self.model.generate_content(full_prompt)
            response_text = response.text
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

    def generate_streaming_response(self, prompt: str, context: str = "", **kwargs):
        import time
        import logging
        full_prompt = self._create_full_prompt(prompt, context)
        try:
            response_iterator = self.model.generate_content(full_prompt, stream=True)
            for chunk in response_iterator:
                if chunk.text:
                    logging.info(f"Received a large chunk from Google: '{chunk.text[:50]}...'")
                    words = chunk.text.split(' ')
                    for i, word in enumerate(words):
                        yield word + (' ' if i < len(words) - 1 else '')
                        time.sleep(0.05)
        except Exception as e:
            raise Exception(f"Google Gemini API streaming error: {str(e)}")

    def _create_full_prompt(self, prompt: str, context: str) -> str:
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

def create_llm_client(provider: str = None) -> LLMClient:
    """Factory function để tạo LLM client"""
    provider = provider or settings.llm_provider
    if provider.lower() == "openai":
        return OpenAIClient()
    elif provider.lower() == "anthropic":
        return AnthropicClient()
    elif provider.lower() == "google":
        return GoogleClient()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

class LLMManager:
    """Manager để quản lý và tạo các LLM client."""

    def __init__(self):
        if not self.get_available_providers():
            raise ValueError("No LLM API keys configured. Please set at least one in the .env file.")

    def generate_response(self, prompt: str, context: str = "", provider: Optional[str] = None, **kwargs) -> tuple[str, dict]:
        provider_to_use = provider or settings.default_provider
        try:
            client = create_llm_client(provider_to_use)
            return client.generate_response(prompt, context, **kwargs)
        except Exception as e:
            raise Exception(f"Error with LLM provider '{provider_to_use}': {str(e)}") from e

    def generate_streaming_response(self, prompt: str, context: str = "", provider: Optional[str] = None, **kwargs):
        provider_to_use = provider or settings.default_provider
        try:
            client = create_llm_client(provider_to_use)
            return client.generate_streaming_response(prompt, context, **kwargs)
        except Exception as e:
            raise Exception(f"Error with LLM provider '{provider_to_use}': {str(e)}") from e

    def is_available(self) -> bool:
        return bool(self.get_available_providers())

    def get_available_providers(self) -> list[str]:
        providers = []
        if settings.openai_api_key:
            providers.append("openai")
        if settings.google_api_key:
            providers.append("google")
        if settings.anthropic_api_key:
            providers.append("anthropic")
        return list(set(providers))
