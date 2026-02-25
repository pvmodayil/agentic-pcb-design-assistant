from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path
from typing import TypedDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # Up 3 levels: core -> src -> root

class LLMConfigDict(TypedDict):
    model_name: str
    base_url: str
    temperature: float

class LLMSettings(BaseSettings):
    model_name: str = Field("gemma3", description="LLM model name")
    base_url: str = Field("http://localhost:11434/v1", description="Ollama base URL")
    temperature: float = Field(0.4, description="Generation temperature")
    

def load_settings(key: str) -> LLMSettings:
    """Load YAML config file and initialise LLM model settings"""
    
    # Navigate from src/agents/ to project root, then to config/
    config_path: Path = PROJECT_ROOT / "config" / "llm_config.yaml"
    
    if not config_path.exists():
        return LLMSettings() #type:ignore
    
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Handle nested 'llm:' structure
    llm_config: LLMConfigDict = data.get(key, {})
    return LLMSettings(model_name=llm_config["model_name"], base_url=llm_config["base_url"], temperature=llm_config["temperature"])