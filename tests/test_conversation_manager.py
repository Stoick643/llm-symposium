"""
Tests for ConversationManager class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from conversation_manager import LLMConversation
from models import ConversationTurn, ConversationMetrics
from config_manager import ConversationConfig


class TestLLMConversation:
    """Test cases for LLMConversation class."""
    
    def test_init_default(self, mock_api_keys):
        """Test LLMConversation initialization with default parameters."""
        with patch('conversation_manager.anthropic.Anthropic'), \
             patch('conversation_manager.OpenAI'):
            
            config = ConversationConfig(models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"])
            conversation = LLMConversation(config=config, api_key=mock_api_keys['anthropic'])
            
            assert conversation.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
            assert conversation.config.ai_aware_mode is False
            assert conversation.config.mode == "full"
            assert conversation.config.window_size == 10
            assert conversation.config.template is None
            assert conversation.template_config is None
            assert conversation.config.enable_quality_metrics is False
            assert conversation.conversation_history == []
            assert conversation.total_input_tokens == 0
            assert conversation.total_output_tokens == 0
            assert conversation.context_size_per_turn == []
            assert conversation.response_times == []
    
    def test_init_custom_params(self, mock_api_keys):
        """Test LLMConversation initialization with custom parameters."""
        with patch('conversation_manager.anthropic.Anthropic'), \
             patch('conversation_manager.OpenAI'):
            
            config = ConversationConfig(
                models=["gpt-4", "claude-3-opus-20240229"],
                ai_aware_mode=True,
                mode="sliding",
                window_size=5,
                template="debate",
                enable_quality_metrics=True
            )
            conversation = LLMConversation(config=config, api_key=mock_api_keys['anthropic'])
            
            assert conversation.models == ["gpt-4", "claude-3-opus-20240229"]
            assert conversation.config.ai_aware_mode is True
            assert conversation.config.mode == "sliding"
            assert conversation.config.window_size == 5
            assert conversation.config.template == "debate"
            assert conversation.template_config is not None
            assert conversation.config.enable_quality_metrics is True
    
    def test_get_pricing_claude(self, mock_conversation):
        """Test pricing retrieval for Claude models."""
        pricing = mock_conversation._get_pricing("claude-3-sonnet-20240229")
        assert pricing == {"input": 3.0, "output": 15.0}
    
    def test_get_pricing_openai(self, mock_conversation):
        """Test pricing retrieval for OpenAI models."""
        pricing = mock_conversation._get_pricing("gpt-4")
        assert pricing == {"input": 30.0, "output": 60.0}
        
        pricing = mock_conversation._get_pricing("gpt-3.5-turbo")
        assert pricing == {"input": 0.5, "output": 1.5}
    
    def test_get_pricing_deepseek(self, mock_conversation):
        """Test pricing retrieval for DeepSeek models."""
        pricing = mock_conversation._get_pricing("deepseek-chat")
        assert pricing == {"input": 0.14, "output": 0.28}
    
    def test_get_pricing_unknown(self, mock_conversation):
        """Test pricing retrieval for unknown models defaults to Claude."""
        pricing = mock_conversation._get_pricing("unknown-model")
        assert pricing == {"input": 3.0, "output": 15.0}
    
    def test_build_full_context_empty(self, mock_conversation):
        """Test building full context with empty conversation history."""
        context = mock_conversation._build_full_context("test-model-1")
        assert context == []
    
    def test_build_full_context_with_history(self, mock_conversation):
        """Test building full context with conversation history."""
        # Add some conversation turns
        mock_conversation.conversation_history = [
            ConversationTurn("test-model-1", "Hello", 1234567890.0),
            ConversationTurn("test-model-2", "Hi there", 1234567891.0),
            ConversationTurn("test-model-1", "How are you?", 1234567892.0)
        ]
        
        # Build context for model 1
        context = mock_conversation._build_full_context("test-model-1")
        expected = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "How are you?"}
        ]
        assert context == expected
        
        # Build context for model 2
        context = mock_conversation._build_full_context("test-model-2")
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        assert context == expected
    
    def test_apply_sliding_window_small(self, mock_conversation):
        """Test sliding window with context smaller than window size."""
        context = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        mock_conversation.window_size = 5
        result = mock_conversation._apply_sliding_window(context)
        assert result == context
    
    def test_apply_sliding_window_large(self, mock_conversation):
        """Test sliding window with context larger than window size."""
        context = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
            {"role": "assistant", "content": "6"}
        ]
        
        mock_conversation.window_size = 4
        result = mock_conversation._apply_sliding_window(context)
        expected = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "5"},
            {"role": "assistant", "content": "6"}
        ]
        assert result == expected
    
    def test_build_context_for_model_full_mode(self, mock_conversation):
        """Test building context for full mode."""
        mock_conversation.mode = "full"
        mock_conversation.conversation_history = [
            ConversationTurn("test-model-1", "Hello", 1234567890.0),
            ConversationTurn("test-model-2", "Hi there", 1234567891.0)
        ]
        
        context = mock_conversation._build_context_for_model("test-model-1")
        expected = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hi there"}
        ]
        assert context == expected
    
    def test_build_context_for_model_sliding_mode(self, mock_conversation):
        """Test building context for sliding mode."""
        mock_conversation.mode = "sliding"
        mock_conversation.window_size = 2
        mock_conversation.conversation_history = [
            ConversationTurn("test-model-1", "1", 1234567890.0),
            ConversationTurn("test-model-2", "2", 1234567891.0),
            ConversationTurn("test-model-1", "3", 1234567892.0),
            ConversationTurn("test-model-2", "4", 1234567893.0)
        ]
        
        context = mock_conversation._build_context_for_model("test-model-1")
        # Should get last 2 messages only
        expected = [
            {"role": "assistant", "content": "3"},
            {"role": "user", "content": "4"}
        ]
        assert context == expected
    
    def test_print_metrics_summary(self, mock_conversation, capsys):
        """Test printing metrics summary."""
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
        
        mock_conversation._print_metrics_summary(metrics)
        
        captured = capsys.readouterr()
        assert "CONVERSATION METRICS" in captured.out
        assert "Mode: full" in captured.out
        assert "Turns Completed: 3" in captured.out
        assert "Total Input Tokens: 100" in captured.out
        assert "Total Output Tokens: 200" in captured.out
        assert "Total Cost: $0.0010" in captured.out
        assert "Average Response Time: 1.5s" in captured.out
        assert "Average Context Size: 20 tokens" in captured.out
    
    def test_print_conversation_summary(self, mock_conversation, capsys):
        """Test printing conversation summary."""
        mock_conversation.conversation_history = [
            ConversationTurn("test-model-1", "Hello", 1234567890.0),
            ConversationTurn("test-model-2", "Hi there", 1234567891.0)
        ]
        
        mock_conversation.print_conversation_summary()
        
        captured = capsys.readouterr()
        assert "CONVERSATION SUMMARY" in captured.out
        assert "Model 1 (test-model-1):" in captured.out
        assert "Hello" in captured.out
        assert "Model 2 (test-model-2):" in captured.out
        assert "Hi there" in captured.out
    
    def test_call_model_routing_claude(self, mock_conversation):
        """Test model call routing for Claude models."""
        result = mock_conversation._call_model("claude-3-sonnet-20240229", "test prompt", [])
        assert result == ("Test response", 50, 100, 1.0)
        mock_conversation._call_anthropic_model.assert_called_once()
    
    def test_call_model_routing_openai(self, mock_conversation):
        """Test model call routing for OpenAI models."""
        result = mock_conversation._call_model("gpt-4", "test prompt", [])
        assert result == ("Test response", 50, 100, 1.0)
        mock_conversation._call_openai_compatible_model.assert_called_once()
    
    def test_call_model_routing_deepseek(self, mock_conversation):
        """Test model call routing for DeepSeek models."""
        result = mock_conversation._call_model("deepseek-chat", "test prompt", [])
        assert result == ("Test response", 50, 100, 1.0)
        mock_conversation._call_openai_compatible_model.assert_called_once()
    
    def test_call_model_with_template(self, mock_api_keys):
        """Test model call with template system prompt."""
        with patch('conversation_manager.anthropic.Anthropic'), \
             patch('conversation_manager.OpenAI'):
            
            config = ConversationConfig(
                models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
                template="debate"
            )
            conversation = LLMConversation(
                config=config,
                api_key=mock_api_keys['anthropic']
            )
            conversation._call_anthropic_model = Mock(return_value=("Test response", 50, 100, 1.0))
            
            result = conversation._call_model("claude-3-sonnet-20240229", "test prompt", [])
            
            # Check that the call was made
            assert result == ("Test response", 50, 100, 1.0)
            
            # Check that system prompt was added
            call_args = conversation._call_anthropic_model.call_args
            messages = call_args[0][1]  # Second argument is messages
            assert len(messages) >= 1
            assert messages[0]["role"] == "system"
            assert "debate" in messages[0]["content"].lower()
    
    def test_call_model_with_ai_aware_mode(self, mock_api_keys):
        """Test model call with AI-aware mode."""
        with patch('conversation_manager.anthropic.Anthropic'), \
             patch('conversation_manager.OpenAI'):
            
            config = ConversationConfig(
                models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
                ai_aware_mode=True
            )
            conversation = LLMConversation(
                config=config,
                api_key=mock_api_keys['anthropic']
            )
            conversation._call_anthropic_model = Mock(return_value=("Test response", 50, 100, 1.0))
            
            result = conversation._call_model("claude-3-sonnet-20240229", "test prompt", [])
            
            # Check that the call was made
            assert result == ("Test response", 50, 100, 1.0)
            
            # Check that AI-aware system prompt was added
            call_args = conversation._call_anthropic_model.call_args
            messages = call_args[0][1]  # Second argument is messages
            assert len(messages) >= 1
            assert messages[0]["role"] == "system"
            assert "AI language model" in messages[0]["content"]