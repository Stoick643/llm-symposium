"""
Pytest configuration and fixtures for LLM Talk tests.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch

from template_manager import TemplateManager
from conversation_manager import LLMConversation
from models import ConversationTurn, ConversationMetrics
from config_manager import ConversationConfig


@pytest.fixture
def template_manager():
    """Create a TemplateManager instance for testing."""
    return TemplateManager()


@pytest.fixture
def mock_api_keys():
    """Mock API keys for testing."""
    return {
        'anthropic': 'test-anthropic-key',
        'openai': 'test-openai-key',
        'deepseek': 'test-deepseek-key'
    }


@pytest.fixture
def conversation_turn():
    """Create a sample conversation turn."""
    return ConversationTurn(
        speaker="test-model",
        message="Test message",
        timestamp=1234567890.0
    )


@pytest.fixture
def conversation_metrics():
    """Create sample conversation metrics."""
    return ConversationMetrics(
        mode="full",
        window_size=None,
        total_input_tokens=100,
        total_output_tokens=200,
        total_cost=0.001,
        context_size_per_turn=[10, 20, 30],
        response_times=[1.0, 1.5, 2.0],
        turns_completed=3
    )


@pytest.fixture
def mock_conversation():
    """Create a mock conversation manager with mocked API calls."""
    with patch('conversation_manager.anthropic.Anthropic'), \
         patch('conversation_manager.OpenAI'):
        
        config = ConversationConfig(
            models=["test-model-1", "test-model-2"],
            mode="full",
            turns=10
        )
        
        conversation = LLMConversation(
            config=config,
            api_key="test-key"
        )
        
        # Mock the API call methods
        conversation._call_anthropic_model = Mock(return_value=("Test response", 50, 100, 1.0))
        conversation._call_openai_compatible_model = Mock(return_value=("Test response", 50, 100, 1.0))
        
        return conversation


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Mock environment variables for all tests."""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'DEEPSEEK_API_KEY': 'test-deepseek-key'
    }):
        yield