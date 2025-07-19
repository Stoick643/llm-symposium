"""
Tests for data models.
"""

import pytest
from models import (
    ConversationTurn, ConversationMetrics,
    Provider, ProviderInfo, ModelInfo, ModelManager,
    model_manager
)


class TestConversationTurn:
    """Test cases for ConversationTurn dataclass."""
    
    def test_init(self):
        """Test ConversationTurn initialization."""
        turn = ConversationTurn(
            speaker="test-model",
            message="Test message",
            timestamp=1234567890.0
        )
        
        assert turn.speaker == "test-model"
        assert turn.message == "Test message"
        assert turn.timestamp == 1234567890.0
    
    def test_types(self):
        """Test that ConversationTurn has correct types."""
        turn = ConversationTurn(
            speaker="test-model",
            message="Test message",
            timestamp=1234567890.0
        )
        
        assert isinstance(turn.speaker, str)
        assert isinstance(turn.message, str)
        assert isinstance(turn.timestamp, float)


class TestConversationMetrics:
    """Test cases for ConversationMetrics dataclass."""
    
    def test_init_required_only(self):
        """Test ConversationMetrics initialization with required fields only."""
        metrics = ConversationMetrics(
            mode="full",
            window_size=None,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.001,
            context_size_per_turn=[10, 20, 30],
            response_times=[1.0, 1.5, 2.0],
            turns_completed=3
        )
        
        assert metrics.mode == "full"
        assert metrics.window_size is None
        assert metrics.total_input_tokens == 100
        assert metrics.total_output_tokens == 200
        assert metrics.total_cost == 0.001
        assert metrics.context_size_per_turn == [10, 20, 30]
        assert metrics.response_times == [1.0, 1.5, 2.0]
        assert metrics.turns_completed == 3
        
        # Optional fields should be None by default
        assert metrics.coherence_scores is None
        assert metrics.topic_drift_scores is None
        assert metrics.repetition_rates is None
        assert metrics.quality_summary is None
    
    def test_init_with_optional_fields(self):
        """Test ConversationMetrics initialization with optional fields."""
        metrics = ConversationMetrics(
            mode="sliding",
            window_size=10,
            total_input_tokens=150,
            total_output_tokens=250,
            total_cost=0.002,
            context_size_per_turn=[5, 15, 25],
            response_times=[0.8, 1.2, 1.8],
            turns_completed=5,
            coherence_scores=[0.8, 0.9, 0.7],
            topic_drift_scores=[0.1, 0.2, 0.3],
            repetition_rates=[0.05, 0.10, 0.15],
            quality_summary={"avg_coherence": 0.8}
        )
        
        assert metrics.mode == "sliding"
        assert metrics.window_size == 10
        assert metrics.coherence_scores == [0.8, 0.9, 0.7]
        assert metrics.topic_drift_scores == [0.1, 0.2, 0.3]
        assert metrics.repetition_rates == [0.05, 0.10, 0.15]
        assert metrics.quality_summary == {"avg_coherence": 0.8}
    
    def test_types(self):
        """Test that ConversationMetrics has correct types."""
        metrics = ConversationMetrics(
            mode="full",
            window_size=10,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.001,
            context_size_per_turn=[10, 20, 30],
            response_times=[1.0, 1.5, 2.0],
            turns_completed=3
        )
        
        assert isinstance(metrics.mode, str)
        assert isinstance(metrics.window_size, int)
        assert isinstance(metrics.total_input_tokens, int)
        assert isinstance(metrics.total_output_tokens, int)
        assert isinstance(metrics.total_cost, float)
        assert isinstance(metrics.context_size_per_turn, list)
        assert isinstance(metrics.response_times, list)
        assert isinstance(metrics.turns_completed, int)
    
    def test_valid_modes(self):
        """Test that valid conversation modes are accepted."""
        valid_modes = ["full", "sliding", "cache", "sliding_cache"]
        
        for mode in valid_modes:
            metrics = ConversationMetrics(
                mode=mode,
                window_size=None,
                total_input_tokens=100,
                total_output_tokens=200,
                total_cost=0.001,
                context_size_per_turn=[10, 20, 30],
                response_times=[1.0, 1.5, 2.0],
                turns_completed=3
            )
            assert metrics.mode == mode
    
    def test_list_types(self):
        """Test that list fields contain correct types."""
        metrics = ConversationMetrics(
            mode="full",
            window_size=None,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.001,
            context_size_per_turn=[10, 20, 30],
            response_times=[1.0, 1.5, 2.0],
            turns_completed=3
        )
        
        # Check context_size_per_turn contains integers
        for size in metrics.context_size_per_turn:
            assert isinstance(size, int)
        
        # Check response_times contains floats
        for time in metrics.response_times:
            assert isinstance(time, float)
    
    def test_quality_metrics_optional(self):
        """Test that quality metrics are properly optional."""
        # Without quality metrics
        metrics1 = ConversationMetrics(
            mode="full",
            window_size=None,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.001,
            context_size_per_turn=[10, 20, 30],
            response_times=[1.0, 1.5, 2.0],
            turns_completed=3
        )
        
        assert metrics1.coherence_scores is None
        assert metrics1.topic_drift_scores is None
        assert metrics1.repetition_rates is None
        assert metrics1.quality_summary is None
        
        # With quality metrics
        metrics2 = ConversationMetrics(
            mode="full",
            window_size=None,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.001,
            context_size_per_turn=[10, 20, 30],
            response_times=[1.0, 1.5, 2.0],
            turns_completed=3,
            coherence_scores=[0.8, 0.9],
            topic_drift_scores=[0.1, 0.2],
            repetition_rates=[0.05, 0.10],
            quality_summary={"test": "data"}
        )
        
        assert metrics2.coherence_scores == [0.8, 0.9]
        assert metrics2.topic_drift_scores == [0.1, 0.2]
        assert metrics2.repetition_rates == [0.05, 0.10]
        assert metrics2.quality_summary == {"test": "data"}


class TestProvider:
    """Test cases for Provider enum."""
    
    def test_provider_enum_values(self):
        """Test that Provider enum has expected values."""
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.OPENAI.value == "openai"
        assert Provider.DEEPSEEK.value == "deepseek"
        assert Provider.MOONSHOT.value == "moonshot"
    
    def test_provider_enum_completeness(self):
        """Test that all expected providers are present."""
        provider_values = {p.value for p in Provider}
        expected = {"anthropic", "openai", "deepseek", "moonshot"}
        assert provider_values == expected


class TestProviderInfo:
    """Test cases for ProviderInfo dataclass."""
    
    def test_provider_info_creation(self):
        """Test ProviderInfo creation with all fields."""
        provider_info = ProviderInfo(
            name="test-provider",
            provider=Provider.ANTHROPIC,
            display_name="Test Provider",
            description="Test provider for testing",
            base_url="https://api.test.com",
            api_key_env="TEST_API_KEY",
            client_type="anthropic"
        )
        
        assert provider_info.name == "test-provider"
        assert provider_info.provider == Provider.ANTHROPIC
        assert provider_info.display_name == "Test Provider"
        assert provider_info.description == "Test provider for testing"
        assert provider_info.base_url == "https://api.test.com"
        assert provider_info.api_key_env == "TEST_API_KEY"
        assert provider_info.client_type == "anthropic"
    
    def test_provider_info_optional_fields(self):
        """Test ProviderInfo with optional fields."""
        provider_info = ProviderInfo(
            name="minimal-provider",
            provider=Provider.OPENAI,
            display_name="Minimal Provider",
            description="Minimal test provider",
            api_key_env="MIN_API_KEY"
            # base_url and client_type will use defaults
        )
        
        assert provider_info.name == "minimal-provider"
        assert provider_info.provider == Provider.OPENAI
        assert provider_info.display_name == "Minimal Provider"
        assert provider_info.description == "Minimal test provider"
        assert provider_info.api_key_env == "MIN_API_KEY"
        assert provider_info.base_url is None  # default
        assert provider_info.client_type == "anthropic"  # default


class TestModelInfo:
    """Test cases for ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test ModelInfo creation with all fields."""
        model_info = ModelInfo(
            name="test-model-v1",
            provider=Provider.ANTHROPIC,
            display_name="Test Model v1",
            description="A test model for testing",
            context_length=128000,
            pricing={"input": 5.0, "output": 15.0},
            capabilities=["reasoning", "coding", "analysis"]
        )
        
        assert model_info.name == "test-model-v1"
        assert model_info.provider == Provider.ANTHROPIC
        assert model_info.display_name == "Test Model v1"
        assert model_info.description == "A test model for testing"
        assert model_info.context_length == 128000
        assert model_info.pricing == {"input": 5.0, "output": 15.0}
        assert model_info.capabilities == ["reasoning", "coding", "analysis"]
    
    def test_model_info_required_fields_only(self):
        """Test ModelInfo with only required fields."""
        model_info = ModelInfo(
            name="minimal-model",
            provider=Provider.OPENAI,
            display_name="Minimal Model",
            description="Minimal test model",
            context_length=4096,
            pricing={"input": 2.0, "output": 6.0}
            # capabilities is optional
        )
        
        assert model_info.name == "minimal-model"
        assert model_info.provider == Provider.OPENAI
        assert model_info.display_name == "Minimal Model"
        assert model_info.description == "Minimal test model"
        assert model_info.context_length == 4096
        assert model_info.pricing == {"input": 2.0, "output": 6.0}
        # Optional field should have default (empty list, not None)
        assert model_info.capabilities == []
    
    def test_model_info_get_pricing(self):
        """Test getting pricing dictionary from ModelInfo."""
        model_info = ModelInfo(
            name="pricing-test",
            provider=Provider.ANTHROPIC,
            display_name="Pricing Test Model",
            description="Test model for pricing",
            context_length=8192,
            pricing={"input": 3.0, "output": 9.0}
        )
        
        assert model_info.pricing["input"] == 3.0
        assert model_info.pricing["output"] == 9.0


class TestModelManager:
    """Test cases for ModelManager class."""
    
    @pytest.fixture
    def test_manager(self):
        """Create a test ModelManager with sample data."""
        return ModelManager()
    
    def test_model_manager_initialization(self, test_manager):
        """Test that ModelManager initializes with expected data."""
        assert isinstance(test_manager._providers, dict)
        assert isinstance(test_manager._models, dict)
        
        # Check that some expected providers exist
        assert Provider.ANTHROPIC in test_manager._providers
        assert Provider.OPENAI in test_manager._providers
        assert Provider.DEEPSEEK in test_manager._providers
        assert Provider.MOONSHOT in test_manager._providers
        
        # Check that some expected models exist
        assert "claude-3-sonnet-20240229" in test_manager._models
        assert "gpt-4" in test_manager._models
        assert "deepseek-chat" in test_manager._models
        assert "kimi-k2-0711-preview" in test_manager._models
    
    def test_get_model_existing(self, test_manager):
        """Test getting info for existing model."""
        model_info = test_manager.get_model("claude-3-sonnet-20240229")
        
        assert model_info is not None
        assert model_info.name == "claude-3-sonnet-20240229"
        assert model_info.provider == Provider.ANTHROPIC
        assert "input" in model_info.pricing
        assert "output" in model_info.pricing
    
    def test_get_model_nonexistent(self, test_manager):
        """Test getting info for nonexistent model."""
        model_info = test_manager.get_model("nonexistent-model")
        assert model_info is None
    
    def test_get_provider_for_model_existing(self, test_manager):
        """Test getting provider for existing model."""
        provider = test_manager.get_provider_for_model("claude-3-sonnet-20240229")
        assert provider == Provider.ANTHROPIC
        
        provider = test_manager.get_provider_for_model("gpt-4")
        assert provider == Provider.OPENAI
        
        provider = test_manager.get_provider_for_model("deepseek-chat")
        assert provider == Provider.DEEPSEEK
        
        provider = test_manager.get_provider_for_model("kimi-k2-0711-preview")
        assert provider == Provider.MOONSHOT
    
    def test_get_provider_for_model_nonexistent(self, test_manager):
        """Test getting provider for nonexistent model."""
        provider = test_manager.get_provider_for_model("nonexistent-model")
        assert provider is None
    
    def test_get_pricing_existing(self, test_manager):
        """Test getting pricing for existing model."""
        pricing = test_manager.get_pricing("claude-3-sonnet-20240229")
        
        assert pricing is not None
        assert "input" in pricing
        assert "output" in pricing
        assert isinstance(pricing["input"], (int, float))
        assert isinstance(pricing["output"], (int, float))
        assert pricing["input"] > 0
        assert pricing["output"] > 0
    
    def test_get_pricing_nonexistent(self, test_manager):
        """Test getting pricing for nonexistent model."""
        pricing = test_manager.get_pricing("nonexistent-model")
        assert pricing is None
    
    def test_get_model_names_by_provider(self, test_manager):
        """Test listing model names by provider."""
        anthropic_models = test_manager.get_model_names_by_provider(Provider.ANTHROPIC)
        assert isinstance(anthropic_models, list)
        assert len(anthropic_models) > 0
        assert "claude-3-sonnet-20240229" in anthropic_models
        
        openai_models = test_manager.get_model_names_by_provider(Provider.OPENAI)
        assert isinstance(openai_models, list)
        assert "gpt-4" in openai_models
    
    def test_get_model_names(self, test_manager):
        """Test getting all model names."""
        all_models = test_manager.get_model_names()
        
        assert isinstance(all_models, list)
        assert len(all_models) > 0
        assert "claude-3-sonnet-20240229" in all_models
        assert "gpt-4" in all_models
        assert "deepseek-chat" in all_models
        assert "kimi-k2-0711-preview" in all_models
    
    def test_list_models_formatted(self, test_manager):
        """Test formatted model listing."""
        formatted = test_manager.list_models_formatted()
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "claude-3-sonnet-20240229" in formatted
        assert "Anthropic" in formatted
        assert "OpenAI" in formatted
        assert "DeepSeek" in formatted
        assert "Moonshot" in formatted
    
    def test_is_valid_model_existing(self, test_manager):
        """Test model validation for existing models."""
        assert test_manager.is_valid_model("claude-3-sonnet-20240229") is True
        assert test_manager.is_valid_model("gpt-4") is True
        assert test_manager.is_valid_model("deepseek-chat") is True
        assert test_manager.is_valid_model("kimi-k2-0711-preview") is True
    
    def test_is_valid_model_nonexistent(self, test_manager):
        """Test model validation for nonexistent models."""
        assert test_manager.is_valid_model("nonexistent-model") is False
        assert test_manager.is_valid_model("") is False
        assert test_manager.is_valid_model(None) is False
    
    def test_get_provider_info(self, test_manager):
        """Test getting provider information."""
        anthropic_info = test_manager.get_provider_info(Provider.ANTHROPIC)
        assert anthropic_info is not None
        assert anthropic_info.display_name == "Anthropic"
        assert anthropic_info.api_key_env == "ANTHROPIC_API_KEY"
        
        openai_info = test_manager.get_provider_info(Provider.OPENAI)
        assert openai_info is not None
        assert openai_info.display_name == "OpenAI"
        assert openai_info.api_key_env == "OPENAI_API_KEY"
    
    def test_model_manager_singleton(self):
        """Test that the global model_manager is properly initialized."""
        assert model_manager is not None
        assert isinstance(model_manager, ModelManager)
        
        # Test some basic functionality
        assert "claude-3-sonnet-20240229" in model_manager._models
        assert model_manager.get_provider_for_model("claude-3-sonnet-20240229") == Provider.ANTHROPIC


class TestModelManagerIntegration:
    """Integration tests for ModelManager with real data."""
    
    def test_all_models_have_valid_providers(self):
        """Test that all models have valid provider assignments."""
        for model_name, model_info in model_manager._models.items():
            assert model_info.provider in model_manager._providers
            provider_info = model_manager._providers[model_info.provider]
            assert isinstance(provider_info, ProviderInfo)
    
    def test_all_models_have_pricing(self):
        """Test that all models have valid pricing information."""
        for model_name, model_info in model_manager._models.items():
            assert model_info.pricing["input"] > 0
            assert model_info.pricing["output"] > 0
            
            pricing = model_manager.get_pricing(model_name)
            assert pricing is not None
            assert pricing["input"] == model_info.pricing["input"]
            assert pricing["output"] == model_info.pricing["output"]
    
    def test_provider_model_consistency(self):
        """Test consistency between providers and models."""
        # Get all providers that have models
        providers_with_models = set()
        for model_info in model_manager._models.values():
            providers_with_models.add(model_info.provider)
        
        # All providers with models should be in the providers registry
        for provider in providers_with_models:
            assert provider in model_manager._providers
    
    def test_moonshot_integration(self):
        """Test Moonshot provider integration specifically."""
        # Check that Moonshot provider exists
        assert Provider.MOONSHOT in model_manager._providers
        moonshot_info = model_manager._providers[Provider.MOONSHOT]
        assert moonshot_info.display_name == "Moonshot"
        assert moonshot_info.api_key_env == "MOONSHOT_API_KEY"
        
        # Check that kimi model exists and is correctly configured
        assert "kimi-k2-0711-preview" in model_manager._models
        kimi_model = model_manager._models["kimi-k2-0711-preview"]
        assert kimi_model.provider == Provider.MOONSHOT
        assert kimi_model.pricing["input"] > 0
        assert kimi_model.pricing["output"] > 0
    
    def test_model_names_are_unique(self):
        """Test that all model names are unique."""
        model_names = list(model_manager._models.keys())
        assert len(model_names) == len(set(model_names))
    
    def test_provider_names_are_unique(self):
        """Test that all provider names are unique."""
        provider_names = [info.display_name for info in model_manager._providers.values()]
        assert len(provider_names) == len(set(provider_names))