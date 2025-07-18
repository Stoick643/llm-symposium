"""
Tests for data models.
"""

import pytest
from models import ConversationTurn, ConversationMetrics


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