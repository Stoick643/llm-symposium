"""
Test suite for database_service.py

Tests the DatabaseService class and utility functions for conversation persistence.
"""

import pytest
import tempfile
import os
from flask import Flask
from datetime import datetime

from database_service import DatabaseService, create_conversation, save_turn, update_metrics
from database_models import db, Conversation, Turn


class TestDatabaseService:
    """Test the DatabaseService class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        yield f'sqlite:///{db_path}'
        os.unlink(db_path)
    
    @pytest.fixture
    def service(self, temp_db):
        """Create a DatabaseService instance with temporary database."""
        service = DatabaseService(database_url=temp_db)
        return service
    
    @pytest.fixture
    def sample_models(self):
        """Sample model list for testing."""
        return ["claude-3-sonnet-20240229", "gpt-4"]
    
    def test_init_creates_tables(self, temp_db):
        """Test that DatabaseService initialization creates tables."""
        service = DatabaseService(database_url=temp_db)
        
        with service.app.app_context():
            # Check that tables exist by trying to query them
            assert Conversation.query.count() == 0
            assert Turn.query.count() == 0
    
    def test_create_conversation_success(self, service, sample_models):
        """Test successful conversation creation."""
        conversation_id = service.create_conversation(
            models=sample_models,
            initial_prompt="Test prompt",
            mode="full",
            window_size=10,
            template="debate",
            ai_aware_mode=True
        )
        
        assert conversation_id is not None
        assert isinstance(conversation_id, int)
        
        # Verify conversation was created correctly
        with service.app.app_context():
            conv = Conversation.query.get(conversation_id)
            assert conv is not None
            assert conv.get_models() == sample_models
            assert conv.initial_prompt == "Test prompt"
            assert conv.mode == "full"
            assert conv.window_size == 10
            assert conv.template == "debate"
            assert conv.ai_aware_mode is True
            assert conv.status == "active"
    
    def test_create_conversation_with_minimal_args(self, service, sample_models):
        """Test conversation creation with minimal arguments."""
        conversation_id = service.create_conversation(
            models=sample_models,
            initial_prompt="Minimal test"
        )
        
        assert conversation_id is not None
        
        with service.app.app_context():
            conv = Conversation.query.get(conversation_id)
            assert conv.mode == "full"  # default
            assert conv.window_size == 10  # default
            assert conv.template is None
            assert conv.ai_aware_mode is False  # default
    
    def test_save_turn_success(self, service, sample_models):
        """Test successful turn saving."""
        # Create conversation first
        conversation_id = service.create_conversation(
            models=sample_models,
            initial_prompt="Test prompt"
        )
        
        # Save a turn
        success = service.save_turn(
            conversation_id=conversation_id,
            turn_number=1,
            speaker="claude-3-sonnet-20240229",
            message="Test response",
            input_tokens=50,
            output_tokens=100,
            response_time=1.5,
            context_size=3
        )
        
        assert success is True
        
        # Verify turn was saved correctly
        with service.app.app_context():
            turn = Turn.query.filter_by(conversation_id=conversation_id).first()
            assert turn is not None
            assert turn.turn_number == 1
            assert turn.speaker == "claude-3-sonnet-20240229"
            assert turn.message == "Test response"
            assert turn.input_tokens == 50
            assert turn.output_tokens == 100
            assert turn.response_time == 1.5
            assert turn.context_size == 3
    
    def test_save_turn_nonexistent_conversation(self, service):
        """Test saving turn to nonexistent conversation."""
        success = service.save_turn(
            conversation_id=999,
            turn_number=1,
            speaker="test-model",
            message="Test message"
        )
        
        # Should still succeed (foreign key constraint may not be enforced in SQLite)
        assert success is True
    
    def test_update_conversation_metrics_success(self, service, sample_models):
        """Test successful metrics update."""
        # Create conversation
        conversation_id = service.create_conversation(
            models=sample_models,
            initial_prompt="Test prompt"
        )
        
        # Update metrics
        success = service.update_conversation_metrics(
            conversation_id=conversation_id,
            total_turns=5,
            total_cost=0.025,
            total_input_tokens=250,
            total_output_tokens=500,
            avg_response_time=1.2,
            status="completed"
        )
        
        assert success is True
        
        # Verify metrics were updated
        with service.app.app_context():
            conv = Conversation.query.get(conversation_id)
            assert conv.total_turns == 5
            assert float(conv.total_cost) == 0.025
            assert conv.total_input_tokens == 250
            assert conv.total_output_tokens == 500
            assert float(conv.avg_response_time) == 1.2
            assert conv.status == "completed"
    
    def test_update_metrics_nonexistent_conversation(self, service):
        """Test updating metrics for nonexistent conversation."""
        success = service.update_conversation_metrics(
            conversation_id=999,
            total_turns=1,
            total_cost=0.001,
            total_input_tokens=10,
            total_output_tokens=20,
            avg_response_time=1.0
        )
        
        assert success is False
    
    def test_get_conversation_success(self, service, sample_models):
        """Test successful conversation retrieval."""
        # Create conversation
        conversation_id = service.create_conversation(
            models=sample_models,
            initial_prompt="Test prompt"
        )
        
        # Get conversation
        conv_dict = service.get_conversation(conversation_id)
        
        assert conv_dict is not None
        assert conv_dict['id'] == conversation_id
        assert conv_dict['models'] == sample_models
        assert conv_dict['initial_prompt'] == "Test prompt"
        assert 'created_at' in conv_dict
    
    def test_get_conversation_nonexistent(self, service):
        """Test getting nonexistent conversation."""
        conv_dict = service.get_conversation(999)
        assert conv_dict is None
    
    def test_get_recent_conversations(self, service, sample_models):
        """Test getting recent conversations."""
        # Create multiple conversations
        id1 = service.create_conversation(sample_models, "First prompt")
        id2 = service.create_conversation(sample_models, "Second prompt")
        
        # Get recent conversations
        recent = service.get_recent_conversations(limit=5)
        
        assert len(recent) == 2
        # Should be in reverse chronological order (newest first)
        assert recent[0]['id'] == id2
        assert recent[1]['id'] == id1
    
    def test_get_recent_conversations_with_limit(self, service, sample_models):
        """Test getting recent conversations with limit."""
        # Create 3 conversations
        for i in range(3):
            service.create_conversation(sample_models, f"Prompt {i}")
        
        # Get only 2 recent
        recent = service.get_recent_conversations(limit=2)
        
        assert len(recent) == 2


class TestDatabaseUtilityFunctions:
    """Test the utility functions that use the global database service."""
    
    @pytest.fixture(autouse=True)
    def setup_temp_db(self, monkeypatch):
        """Set up temporary database for utility function tests."""
        # Create temporary database
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        
        # Create new service with temp database
        from database_service import db_service
        temp_service = DatabaseService(database_url=f'sqlite:///{db_path}')
        
        # Replace global service
        monkeypatch.setattr('database_service.db_service', temp_service)
        
        yield
        
        # Cleanup
        os.unlink(db_path)
    
    def test_create_conversation_utility(self):
        """Test the create_conversation utility function."""
        models = ["claude-3-sonnet-20240229", "gpt-4"]
        conversation_id = create_conversation(
            models=models,
            initial_prompt="Test prompt",
            mode="sliding",
            window_size=8
        )
        
        assert conversation_id is not None
        assert isinstance(conversation_id, int)
    
    def test_save_turn_utility(self):
        """Test the save_turn utility function."""
        # Create conversation first
        models = ["claude-3-sonnet-20240229", "gpt-4"]
        conversation_id = create_conversation(models, "Test prompt")
        
        # Save turn
        success = save_turn(
            conversation_id=conversation_id,
            turn_number=1,
            speaker="claude-3-sonnet-20240229",
            message="Test response"
        )
        
        assert success is True
    
    def test_update_metrics_utility(self):
        """Test the update_metrics utility function."""
        # Create conversation first
        models = ["claude-3-sonnet-20240229", "gpt-4"]
        conversation_id = create_conversation(models, "Test prompt")
        
        # Update metrics
        success = update_metrics(
            conversation_id=conversation_id,
            total_turns=3,
            total_cost=0.015,
            total_input_tokens=150,
            total_output_tokens=300,
            avg_response_time=1.1,
            status="completed"
        )
        
        assert success is True


class TestDatabaseErrorHandling:
    """Test error handling scenarios."""
    
    def test_create_conversation_with_invalid_models(self):
        """Test conversation creation with invalid models list."""
        # Empty models list
        conversation_id = create_conversation(
            models=[],
            initial_prompt="Test prompt"
        )
        
        # Should fail (empty list violates NOT NULL constraint)
        assert conversation_id is None
    
    def test_create_conversation_with_none_models(self):
        """Test conversation creation with None models."""
        conversation_id = create_conversation(
            models=None,
            initial_prompt="Test prompt"
        )
        
        # Should fail (None violates NOT NULL constraint)
        assert conversation_id is None
    
    def test_save_turn_with_missing_required_fields(self):
        """Test saving turn with missing required fields."""
        # Create conversation first
        conversation_id = create_conversation(
            models=["test-model"],
            initial_prompt="Test prompt"
        )
        
        # Try to save turn without required fields
        success = save_turn(
            conversation_id=conversation_id,
            turn_number=1,
            speaker="",  # Empty speaker
            message=""   # Empty message
        )
        
        # Should still succeed (database constraints may allow empty strings)
        assert success is True