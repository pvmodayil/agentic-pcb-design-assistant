import socket
import http.client

from dotenv import load_dotenv
import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider

import subprocess
from src.core.settings import LLMSettings
from loguru import logger

load_dotenv()

def is_ollama_running() -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    if sock.connect_ex(('localhost', 11434)) == 0:
        sock.close()
        conn = http.client.HTTPConnection("localhost", 11434, timeout=3)
        conn.request("GET", "/")
        res: http.client.HTTPResponse = conn.getresponse()
        data: bytes = res.read()
        conn.close()
        return res.status == 200 and b"Ollama is running" in data
    sock.close()
    return False

def get_llm_model(llm_settings: LLMSettings) -> OpenAIChatModel | AnthropicModel:
    """
    Get the LLM model based on the provider in settings.
    Supports 'ollama', 'openai', and 'anthropic'.
    """
    
    provider: str = llm_settings.provider.lower()  # add 'provider' field to LLMSettings

    if provider == "ollama":
        if not is_ollama_running():
            print("Starting ollama")
            subprocess.Popen(["ollama", "serve"])
            logger.info("Ollama server started")
        logger.info("Ollama server running")
        return OpenAIChatModel(
            model_name=llm_settings.model_name,
            provider=OllamaProvider(base_url=llm_settings.base_url)
        )

    elif provider == "openai":
        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        logger.info("Using OpenAI model: %s", llm_settings.model_name)
        return OpenAIChatModel(
            model_name=llm_settings.model_name,
            provider=OpenAIProvider(api_key=api_key)
        )

    elif provider == "anthropic":
        api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        logger.info("Using Anthropic model: %s", llm_settings.model_name)
        return AnthropicModel(
            model_name=llm_settings.model_name,
            provider=AnthropicProvider(api_key=api_key)
        )

    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose from: ollama, openai, anthropic")