"""
Data Models for LLM Conversations

Contains dataclasses for conversation turns, metrics, and related data structures.
Also includes central model management system.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    speaker: str
    message: str
    timestamp: float


@dataclass
class ConversationMetrics:
    """Metrics collected during a conversation."""
    mode: str
    window_size: Optional[int]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    context_size_per_turn: List[int]
    response_times: List[float]
    turns_completed: int
    # Quality metrics (optional)
    coherence_scores: Optional[List[float]] = None
    topic_drift_scores: Optional[List[float]] = None
    repetition_rates: Optional[List[float]] = None
    quality_summary: Optional[Dict] = None


class Provider(Enum):
    """LLM API providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"


@dataclass
class ProviderInfo:
    """Information about an LLM provider."""
    name: str
    provider: Provider
    display_name: str
    description: str
    base_url: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    client_type: str = "anthropic"  # "anthropic" or "openai"


@dataclass
class ModelInfo:
    """Information about a specific LLM model."""
    name: str
    provider: Provider
    display_name: str
    description: str
    context_length: int
    pricing: Dict[str, float]  # per million tokens
    capabilities: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class ModelManager:
    """Central model management system."""
    
    def __init__(self):
        self._providers = self._initialize_providers()
        self._models = self._initialize_models()
    
    def _initialize_providers(self) -> Dict[Provider, ProviderInfo]:
        """Initialize all supported providers."""
        providers = {}
        
        providers[Provider.ANTHROPIC] = ProviderInfo(
            name="anthropic",
            provider=Provider.ANTHROPIC,
            display_name="Anthropic",
            description="Anthropic's Claude models with strong reasoning and safety",
            base_url=None,
            api_key_env="ANTHROPIC_API_KEY",
            client_type="anthropic"
        )
        
        providers[Provider.OPENAI] = ProviderInfo(
            name="openai",
            provider=Provider.OPENAI,
            display_name="OpenAI",
            description="OpenAI's GPT models with strong general capabilities",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            client_type="openai"
        )
        
        providers[Provider.DEEPSEEK] = ProviderInfo(
            name="deepseek",
            provider=Provider.DEEPSEEK,
            display_name="DeepSeek",
            description="DeepSeek's cost-effective models with competitive performance",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            client_type="openai"
        )
        
        providers[Provider.MOONSHOT] = ProviderInfo(
            name="moonshot",
            provider=Provider.MOONSHOT,
            display_name="Moonshot",
            description="Moonshot's Kimi models with multilingual capabilities",
            base_url="https://api.moonshot.ai/v1",
            api_key_env="MOONSHOT_API_KEY",
            client_type="openai"
        )
        
        return providers
    
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize all supported models."""
        models = {}
        
        # Anthropic Claude models
        models["claude-3-sonnet-20240229"] = ModelInfo(
            name="claude-3-sonnet-20240229",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3 Sonnet",
            description="Anthropic's balanced model with good performance and speed",
            context_length=200000,
            pricing={"input": 3.0, "output": 15.0},
            capabilities=["reasoning", "coding", "analysis", "creative"]
        )
        
        models["claude-3-haiku-20240307"] = ModelInfo(
            name="claude-3-haiku-20240307",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3 Haiku",
            description="Anthropic's fastest model for quick responses",
            context_length=200000,
            pricing={"input": 0.25, "output": 1.25},
            capabilities=["reasoning", "coding", "analysis"]
        )
        
        models["claude-3-opus-20240229"] = ModelInfo(
            name="claude-3-opus-20240229",
            provider=Provider.ANTHROPIC,
            display_name="Claude 3 Opus",
            description="Anthropic's most capable model for complex tasks",
            context_length=200000,
            pricing={"input": 15.0, "output": 75.0},
            capabilities=["reasoning", "coding", "analysis", "creative", "research"]
        )
        
        # OpenAI models
        models["gpt-4"] = ModelInfo(
            name="gpt-4",
            provider=Provider.OPENAI,
            display_name="GPT-4",
            description="OpenAI's flagship model with strong reasoning abilities",
            context_length=128000,
            pricing={"input": 30.0, "output": 60.0},
            capabilities=["reasoning", "coding", "analysis", "creative"]
        )
        
        models["gpt-4o"] = ModelInfo(
            name="gpt-4o",
            provider=Provider.OPENAI,
            display_name="GPT-4o",
            description="OpenAI's optimized model with improved speed and cost",
            context_length=128000,
            pricing={"input": 2.5, "output": 10.0},
            capabilities=["reasoning", "coding", "analysis", "creative", "multimodal"]
        )
        
        models["gpt-3.5-turbo"] = ModelInfo(
            name="gpt-3.5-turbo",
            provider=Provider.OPENAI,
            display_name="GPT-3.5 Turbo",
            description="OpenAI's fast and cost-effective model",
            context_length=16385,
            pricing={"input": 0.5, "output": 1.5},
            capabilities=["reasoning", "coding", "analysis"]
        )
        
        # DeepSeek models
        models["deepseek-chat"] = ModelInfo(
            name="deepseek-chat",
            provider=Provider.DEEPSEEK,
            display_name="DeepSeek Chat",
            description="DeepSeek's conversational model with competitive performance",
            context_length=32768,
            pricing={"input": 0.14, "output": 0.28},
            capabilities=["reasoning", "coding", "analysis", "creative"]
        )
        
        models["deepseek-coder"] = ModelInfo(
            name="deepseek-coder",
            provider=Provider.DEEPSEEK,
            display_name="DeepSeek Coder",
            description="DeepSeek's specialized coding model",
            context_length=32768,
            pricing={"input": 0.14, "output": 0.28},
            capabilities=["coding", "analysis", "debugging"]
        )
        
        # Moonshot models
        models["kimi-k2-0711-preview"] = ModelInfo(
            name="kimi-k2-0711-preview",
            provider=Provider.MOONSHOT,
            display_name="Kimi K2 Preview",
            description="Moonshot's latest conversational AI model",
            context_length=200000,
            pricing={"input": 2.0, "output": 6.0},  # Estimated pricing
            capabilities=["reasoning", "coding", "analysis", "creative", "multilingual"]
        )
        
        return models
    
    # Provider methods
    def get_provider_info(self, provider: Provider) -> Optional[ProviderInfo]:
        """Get provider info by provider enum."""
        return self._providers.get(provider)
    
    def get_all_providers(self) -> Dict[Provider, ProviderInfo]:
        """Get all available providers."""
        return self._providers.copy()
    
    def get_provider_names(self) -> List[str]:
        """Get list of all provider names."""
        return [provider.value for provider in self._providers.keys()]
    
    # Model methods
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self._models.get(name)
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all available models."""
        return self._models.copy()
    
    def get_models_by_provider(self, provider: Provider) -> Dict[str, ModelInfo]:
        """Get models filtered by provider."""
        return {name: info for name, info in self._models.items() 
                if info.provider == provider}
    
    def get_model_names(self) -> List[str]:
        """Get list of all model names."""
        return list(self._models.keys())
    
    def get_model_names_by_provider(self, provider: Provider) -> List[str]:
        """Get model names filtered by provider."""
        return [name for name, info in self._models.items() 
                if info.provider == provider]
    
    def is_valid_model(self, name: str) -> bool:
        """Check if model name is valid."""
        return name in self._models
    
    def get_pricing(self, name: str) -> Optional[Dict[str, float]]:
        """Get pricing information for a model."""
        model = self.get_model(name)
        return model.pricing if model else None
    
    def get_provider_for_model(self, name: str) -> Optional[Provider]:
        """Get provider for a model."""
        model = self.get_model(name)
        return model.provider if model else None
    
    def get_client_config(self, name: str) -> Optional[Dict[str, str]]:
        """Get client configuration for a model."""
        model = self.get_model(name)
        if not model:
            return None
        
        provider_info = self.get_provider_info(model.provider)
        if not provider_info:
            return None
        
        config = {
            "provider": provider_info.provider.value,
            "client_type": provider_info.client_type,
            "api_key_env": provider_info.api_key_env
        }
        
        if provider_info.base_url:
            config["base_url"] = provider_info.base_url
        
        return config
    
    def list_models_formatted(self) -> str:
        """Get formatted string listing all models."""
        lines = ["Available Models:"]
        lines.append("=" * 60)
        
        for provider in Provider:
            provider_info = self.get_provider_info(provider)
            models = self.get_models_by_provider(provider)
            
            if models and provider_info:
                lines.append(f"\n{provider_info.display_name.upper()}:")
                lines.append(f"  Provider: {provider_info.description}")
                lines.append(f"  API Key: {provider_info.api_key_env}")
                if provider_info.base_url:
                    lines.append(f"  Base URL: {provider_info.base_url}")
                lines.append(f"  Models ({len(models)}):")
                
                for name, info in models.items():
                    pricing = f"${info.pricing['input']:.2f}/${info.pricing['output']:.2f}"
                    lines.append(f"    • {name}")
                    lines.append(f"      {info.display_name} - {info.description}")
                    lines.append(f"      Pricing: {pricing} per 1M tokens (in/out)")
                    lines.append(f"      Context: {info.context_length:,} tokens")
                    if info.capabilities:
                        lines.append(f"      Capabilities: {', '.join(info.capabilities)}")
                    lines.append("")
        
        return "\n".join(lines)


# Global model manager instance
model_manager = ModelManager()