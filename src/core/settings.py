from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path
from typing import NotRequired, TypedDict, Optional
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # Up 2 levels: core -> src -> root

class LLMConfigDict(TypedDict):
    model_name: str
    base_url: NotRequired[str]
    temperature: float

class LLMSettings(BaseSettings):
    provider: str = Field(default="ollama", description="LLM provider")
    model_name: str = Field(default="gemma3", description="LLM model name")
    base_url: str = Field(default="http://localhost:11434/v1", description="Ollama base URL")
    temperature: float = Field(default=0.4, description="Generation temperature")
    

def load_settings(key: str) -> LLMSettings:
    """Load YAML config file and initialise LLM model settings"""
    
    # Navigate from src/agents/ to project root, then to config/
    config_path: Path = PROJECT_ROOT / "config" / "llm_config.yaml"
    
    if not config_path.exists():
        logger.critical("LLM config file missing moving froward with default model = gemma3.")
        return LLMSettings() 
    
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    # Read top-level provider (e.g. "ollama", "anthropic", "openai")
    provider: str = data.get("provider", "ollama").strip().lower()

    # Drill into the active provider block, then the requested key
    provider_block: dict = data.get(provider, {})
    llm_config: LLMConfigDict = provider_block.get(key, {})

    if not llm_config:
        logger.warning(f"No config found for provider={provider}, key={key}. Using defaults with ollama.")
        return LLMSettings() 

    return LLMSettings(
        provider=provider,
        model_name=llm_config["model_name"],
        base_url=llm_config.get("base_url", "http://localhost:11434/v1"),
        temperature=llm_config.get("temperature", 0.2),
    )