"""
SGLang Client - High-level API for LLM inference.

Provides easy-to-use functions for text generation and chat.
"""

import requests
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union
import json


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    stop: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        if self.stop:
            d["stop"] = self.stop
        return d


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    
    @classmethod
    def from_response(cls, response: Dict) -> "GenerationResult":
        usage = response.get("usage", {})
        choices = response.get("choices", [{}])
        
        return cls(
            text=choices[0].get("text", "") if choices else "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choices[0].get("finish_reason", "") if choices else "",
        )


class SGLangClient:
    """
    Client for SGLang server.
    
    Supports both completion and chat endpoints with streaming.
    
    Example:
        >>> client = SGLangClient("http://localhost:30000")
        >>> result = client.generate("What is 2+2?")
        >>> print(result.text)
    """
    
    def __init__(self, url: str = "http://localhost:30000"):
        self.url = url.rstrip("/")
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> Union[GenerationResult, Iterator[str]]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            stream: Whether to stream the response
            
        Returns:
            GenerationResult or iterator of text chunks
        """
        config = config or GenerationConfig()
        
        payload = {
            "text": prompt,
            "stream": stream,
            **config.to_dict(),
        }
        
        if stream:
            return self._stream_generate(payload)
        else:
            response = requests.post(
                f"{self.url}/generate",
                json=payload,
            )
            return GenerationResult.from_response(response.json())
    
    def _stream_generate(self, payload: Dict) -> Iterator[str]:
        """Stream generation response."""
        response = requests.post(
            f"{self.url}/generate",
            json=payload,
            stream=True,
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                if "text" in data:
                    yield data["text"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
    ) -> Union[GenerationResult, Iterator[str]]:
        """
        Chat completion.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            config: Generation configuration
            stream: Whether to stream the response
            
        Returns:
            GenerationResult or iterator of text chunks
        """
        config = config or GenerationConfig()
        
        payload = {
            "messages": messages,
            "stream": stream,
            **config.to_dict(),
        }
        
        if stream:
            return self._stream_chat(payload)
        else:
            response = requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
            )
            result = response.json()
            
            return GenerationResult(
                text=result["choices"][0]["message"]["content"],
                prompt_tokens=result["usage"]["prompt_tokens"],
                completion_tokens=result["usage"]["completion_tokens"],
                total_tokens=result["usage"]["total_tokens"],
                finish_reason=result["choices"][0]["finish_reason"],
            )
    
    def _stream_chat(self, payload: Dict) -> Iterator[str]:
        """Stream chat response."""
        response = requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            stream=True,
        )
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode()
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[GenerationResult]:
        """
        Generate for multiple prompts in batch.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of GenerationResults
        """
        config = config or GenerationConfig()
        
        payload = {
            "text": prompts,
            **config.to_dict(),
        }
        
        response = requests.post(
            f"{self.url}/generate",
            json=payload,
        )
        
        results = response.json()
        return [GenerationResult.from_response({"choices": [r]}) for r in results]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        payload = {"input": texts}
        response = requests.post(f"{self.url}/v1/embeddings", json=payload)
        result = response.json()
        return [d["embedding"] for d in result["data"]]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_client: Optional[SGLangClient] = None


def set_default_url(url: str):
    """Set default server URL."""
    global _default_client
    _default_client = SGLangClient(url)


def _get_client(url: Optional[str] = None) -> SGLangClient:
    """Get client instance."""
    global _default_client
    
    if url:
        return SGLangClient(url)
    
    if _default_client is None:
        _default_client = SGLangClient()
    
    return _default_client


def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    url: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Generate text from a prompt (simple API).
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        url: Optional server URL
        
    Returns:
        Generated text
    """
    client = _get_client(url)
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    result = client.generate(prompt, config)
    return result.text


def generate_batch(
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.7,
    url: Optional[str] = None,
) -> List[str]:
    """Generate text for multiple prompts."""
    client = _get_client(url)
    config = GenerationConfig(max_new_tokens=max_tokens, temperature=temperature)
    results = client.generate_batch(prompts, config)
    return [r.text for r in results]


def chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
    url: Optional[str] = None,
) -> str:
    """
    Chat with the model.
    
    Args:
        messages: Conversation history
        max_tokens: Maximum tokens
        temperature: Sampling temperature
        url: Optional server URL
        
    Returns:
        Assistant's response
    """
    client = _get_client(url)
    config = GenerationConfig(max_new_tokens=max_tokens, temperature=temperature)
    result = client.chat(messages, config)
    return result.text
