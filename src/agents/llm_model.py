import socket
import http.client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
import subprocess
from config.settings import LLMSettings

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

def get_llm_model(llm_settings: LLMSettings) -> OpenAIChatModel:
    """
    Get the LLM model from configurations
    
    :return: ollama model with the preferred model and provider
    :rtype: OpenAIChatModel
    """
    if not is_ollama_running():
        print("Starting ollama")
        subprocess.Popen(["ollama", "serve"])
    ollama_model = OpenAIChatModel(
    model_name=llm_settings.model_name,
    provider=OllamaProvider(base_url=llm_settings.base_url)
    )
    
    return ollama_model