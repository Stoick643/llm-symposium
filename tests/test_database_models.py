"""
Test suite for database_models.py

Tests SQLAlchemy models, JSON field handling, relationships, and query functions.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta
from flask import Flask

from database_models import (
    db, Conversation, Turn, QualityMetric,
    init_database, create_tables,
    get_all_conversations, get_conversation_by_id,
    get_conversations_by_template, search_conversations,
    get_conversation_turns, get_database_stats
)


@pytest.fixture
def app():
    """Create a Flask app with temporary database for testing."""
    app = Flask(__name__)
    
    # Use in-memory SQLite database for faster tests
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['TESTING'] = True
    
    with app.app_context():
        init_database(app)
        create_tables(app)
        yield app


@pytest.fixture
def sample_conversation(app):
    """Create a sample conversation for testing."""
    with app.app_context():
        conv = Conversation(
            initial_prompt="Test prompt for conversation",
            mode="full",
            window_size=10,
            template="debate",
            ai_aware_mode=False,
            status="active",
            total_turns=0,
            total_cost=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            avg_response_time=0.0
        )
        conv.set_models(["claude-3-sonnet-20240229", "gpt-4"])
        
        db.session.add(conv)
        db.session.commit()
        
        return conv


class TestConversationModel:
    """Test the Conversation SQLAlchemy model."""
    
    def test_conversation_creation(self, app):
        """Test basic conversation creation."""
        with app.app_context():
            conv = Conversation(
                initial_prompt="Test prompt",
                mode="sliding",
                window_size=8,
                template="creative",
                ai_aware_mode=True,
                status="active"
            )
            
            db.session.add(conv)
            db.session.commit()
            
            assert conv.id is not None
            assert conv.initial_prompt == "Test prompt"
            assert conv.mode == "sliding"
            assert conv.window_size == 8
            assert conv.template == "creative"
            assert conv.ai_aware_mode is True
            assert conv.status == "active"
            assert conv.created_at is not None
            assert isinstance(conv.created_at, datetime)
    
    def test_models_json_field(self, app):
        """Test the models JSON field functionality."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test")
            
            # Test setting models
            models = ["claude-3-sonnet-20240229", "gpt-4", "deepseek-chat"]
            conv.set_models(models)
            
            assert conv.model_count == 3
            assert conv.get_models() == models
            
            # Test JSON storage
            db.session.add(conv)
            db.session.commit()
            
            # Retrieve and verify
            retrieved = Conversation.query.get(conv.id)
            assert retrieved.get_models() == models
            assert retrieved.model_count == 3
    
    def test_models_empty_list(self, app):
        """Test models field with empty list."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test")
            conv.set_models([])
            
            assert conv.model_count == 0
            assert conv.get_models() == []
    
    def test_models_none_value(self, app):
        """Test models field with None value."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test")
            conv.set_models(None)
            
            assert conv.model_count == 0
            assert conv.get_models() == []
    
    def test_keywords_json_field(self, app):
        """Test the keywords JSON field functionality."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test")
            
            keywords = ["AI", "machine learning", "conversation"]
            conv.set_keywords(keywords)
            
            assert conv.get_keywords() == keywords
            
            # Test persistence
            db.session.add(conv)
            db.session.commit()
            
            retrieved = Conversation.query.get(conv.id)
            assert retrieved.get_keywords() == keywords
    
    def test_keywords_empty_and_none(self, app):
        """Test keywords field with empty list and None."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test")
            
            # Empty list
            conv.set_keywords([])
            assert conv.get_keywords() == []
            
            # None value
            conv.set_keywords(None)
            assert conv.get_keywords() == []
    
    def test_conversation_to_dict(self, app):
        """Test conversation serialization to dictionary."""
        with app.app_context():
            conv = Conversation(
                initial_prompt="Test prompt",
                mode="cache",
                window_size=6,
                template="socratic",
                ai_aware_mode=True,
                status="completed",
                total_turns=5,
                total_cost=0.025,
                total_input_tokens=250,
                total_output_tokens=500,
                avg_response_time=1.2
            )
            conv.set_models(["model1", "model2"])
            conv.set_keywords(["keyword1", "keyword2"])
            
            db.session.add(conv)
            db.session.commit()
            
            conv_dict = conv.to_dict()
            
            # Check all fields are present
            assert conv_dict['id'] == conv.id
            assert conv_dict['initial_prompt'] == "Test prompt"
            assert conv_dict['mode'] == "cache"
            assert conv_dict['window_size'] == 6
            assert conv_dict['template'] == "socratic"
            assert conv_dict['ai_aware_mode'] is True
            assert conv_dict['status'] == "completed"
            assert conv_dict['total_turns'] == 5
            assert conv_dict['total_cost'] == 0.025
            assert conv_dict['total_input_tokens'] == 250
            assert conv_dict['total_output_tokens'] == 500
            assert conv_dict['avg_response_time'] == 1.2
            assert conv_dict['models'] == ["model1", "model2"]
            assert conv_dict['model_count'] == 2
            assert conv_dict['keywords'] == ["keyword1", "keyword2"]
            assert 'created_at' in conv_dict
    
    def test_conversation_repr(self, app):
        """Test conversation string representation."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test prompt")
            conv.set_models(["model1", "model2"])
            
            repr_str = repr(conv)
            assert "model1 vs model2" in repr_str
            assert "Test prompt" in repr_str
    
    def test_conversation_repr_many_models(self, app):
        """Test conversation repr with many models."""
        with app.app_context():
            conv = Conversation(initial_prompt="Test prompt")
            conv.set_models(["model1", "model2", "model3", "model4"])
            
            repr_str = repr(conv)
            assert "model1 vs model2 (+2 more)" in repr_str


class TestTurnModel:
    """Test the Turn SQLAlchemy model."""
    
    def test_turn_creation(self, app, sample_conversation):
        """Test basic turn creation."""
        with app.app_context():
            turn = Turn(
                conversation_id=sample_conversation.id,
                turn_number=1,
                speaker="claude-3-sonnet-20240229",
                message="This is a test response",
                timestamp=datetime.utcnow(),
                input_tokens=50,
                output_tokens=100,
                response_time=1.5,
                context_size=3
            )
            
            db.session.add(turn)
            db.session.commit()
            
            assert turn.id is not None
            assert turn.conversation_id == sample_conversation.id
            assert turn.turn_number == 1
            assert turn.speaker == "claude-3-sonnet-20240229"
            assert turn.message == "This is a test response"
            assert turn.input_tokens == 50
            assert turn.output_tokens == 100
            assert turn.response_time == 1.5
            assert turn.context_size == 3
    
    def test_turn_conversation_relationship(self, app, sample_conversation):
        """Test the relationship between Turn and Conversation."""
        with app.app_context():
            turn = Turn(
                conversation_id=sample_conversation.id,
                turn_number=1,
                speaker="test-model",
                message="Test message"
            )
            
            db.session.add(turn)
            db.session.commit()
            
            # Test relationship from turn to conversation
            assert turn.conversation.id == sample_conversation.id
            
            # Test relationship from conversation to turns
            assert len(sample_conversation.turns) == 1
            assert sample_conversation.turns[0].id == turn.id
    
    def test_turn_to_dict(self, app, sample_conversation):
        """Test turn serialization to dictionary."""
        with app.app_context():
            timestamp = datetime.utcnow()
            turn = Turn(
                conversation_id=sample_conversation.id,
                turn_number=2,
                speaker="gpt-4",
                message="Response message",
                timestamp=timestamp,
                input_tokens=75,
                output_tokens=150,
                response_time=2.1,
                context_size=5
            )
            
            db.session.add(turn)
            db.session.commit()
            
            turn_dict = turn.to_dict()
            
            assert turn_dict['id'] == turn.id
            assert turn_dict['conversation_id'] == sample_conversation.id
            assert turn_dict['turn_number'] == 2
            assert turn_dict['speaker'] == "gpt-4"
            assert turn_dict['message'] == "Response message"
            assert turn_dict['input_tokens'] == 75
            assert turn_dict['output_tokens'] == 150
            assert turn_dict['response_time'] == 2.1
            assert turn_dict['context_size'] == 5
            assert 'timestamp' in turn_dict


class TestQualityMetricModel:
    """Test the QualityMetric SQLAlchemy model."""
    
    def test_quality_metric_creation(self, app, sample_conversation):
        """Test basic quality metric creation."""
        with app.app_context():
            # Create a turn first
            turn = Turn(
                conversation_id=sample_conversation.id,
                turn_number=1,
                speaker="test-model",
                message="Test message"
            )
            db.session.add(turn)
            db.session.commit()
            
            # Create quality metric
            metric = QualityMetric(
                conversation_id=sample_conversation.id,
                turn_id=turn.id,
                coherence_score=0.85,
                relevance_score=0.90,
                engagement_score=0.75,
                overall_score=0.83,
                metadata_json='{"test": "data"}'
            )
            
            db.session.add(metric)
            db.session.commit()
            
            assert metric.id is not None
            assert metric.conversation_id == sample_conversation.id
            assert metric.turn_id == turn.id
            assert metric.coherence_score == 0.85
            assert metric.relevance_score == 0.90
            assert metric.engagement_score == 0.75
            assert metric.overall_score == 0.83
            assert metric.metadata_json == '{"test": "data"}'


class TestDatabaseQueryFunctions:
    """Test the database query utility functions."""
    
    def test_get_all_conversations(self, app):
        """Test getting all conversations with pagination."""
        with app.app_context():
            # Create multiple conversations
            for i in range(5):
                conv = Conversation(initial_prompt=f"Prompt {i}")
                conv.set_models([f"model{i}"])
                db.session.add(conv)
            db.session.commit()
            
            # Test pagination
            result = get_all_conversations(page=1, per_page=3)
            
            assert result.total == 5
            assert len(result.items) == 3
            assert result.pages == 2
            assert result.has_next is True
            assert result.has_prev is False
            
            # Test second page
            result_page2 = get_all_conversations(page=2, per_page=3)
            assert len(result_page2.items) == 2
            assert result_page2.has_next is False
            assert result_page2.has_prev is True
    
    def test_get_conversation_by_id(self, app, sample_conversation):
        """Test getting conversation by ID."""
        with app.app_context():
            # Existing conversation
            conv = get_conversation_by_id(sample_conversation.id)
            assert conv is not None
            assert conv.id == sample_conversation.id
            
            # Non-existent conversation
            conv = get_conversation_by_id(999)
            assert conv is None
    
    def test_get_conversations_by_template(self, app):
        """Test getting conversations by template."""
        with app.app_context():
            # Create conversations with different templates
            conv1 = Conversation(initial_prompt="Prompt 1", template="debate")
            conv2 = Conversation(initial_prompt="Prompt 2", template="debate")
            conv3 = Conversation(initial_prompt="Prompt 3", template="creative")
            
            for conv in [conv1, conv2, conv3]:
                conv.set_models(["test-model"])
                db.session.add(conv)
            db.session.commit()
            
            # Test filtering by template
            debate_convs = get_conversations_by_template("debate")
            assert debate_convs.total == 2
            
            creative_convs = get_conversations_by_template("creative")
            assert creative_convs.total == 1
            
            nonexistent_convs = get_conversations_by_template("nonexistent")
            assert nonexistent_convs.total == 0
    
    def test_search_conversations(self, app):
        """Test searching conversations by content."""
        with app.app_context():
            # Create conversations with different prompts
            conv1 = Conversation(initial_prompt="Machine learning algorithms")
            conv2 = Conversation(initial_prompt="Deep learning neural networks")
            conv3 = Conversation(initial_prompt="Natural language processing")
            
            for conv in [conv1, conv2, conv3]:
                conv.set_models(["test-model"])
                db.session.add(conv)
            db.session.commit()
            
            # Test search
            learning_results = search_conversations("learning")
            assert learning_results.total == 2  # First two contain "learning"
            
            neural_results = search_conversations("neural")
            assert neural_results.total == 1
            
            nonexistent_results = search_conversations("quantum")
            assert nonexistent_results.total == 0
    
    def test_get_conversation_turns(self, app, sample_conversation):
        """Test getting turns for a conversation."""
        with app.app_context():
            # Create multiple turns
            for i in range(3):
                turn = Turn(
                    conversation_id=sample_conversation.id,
                    turn_number=i + 1,
                    speaker=f"model{i % 2}",
                    message=f"Message {i}"
                )
                db.session.add(turn)
            db.session.commit()
            
            turns = get_conversation_turns(sample_conversation.id)
            
            assert len(turns) == 3
            # Should be ordered by turn_number
            assert turns[0].turn_number == 1
            assert turns[1].turn_number == 2
            assert turns[2].turn_number == 3
    
    def test_get_database_stats(self, app):
        """Test getting database statistics."""
        with app.app_context():
            # Create test data
            conv1 = Conversation(
                initial_prompt="Test 1",
                status="completed",
                total_turns=3,
                total_cost=0.015,
                template="debate"
            )
            conv1.set_models(["model1", "model2"])
            
            conv2 = Conversation(
                initial_prompt="Test 2",
                status="active",
                total_turns=5,
                total_cost=0.025,
                template="creative"
            )
            conv2.set_models(["model2", "model3"])
            
            db.session.add_all([conv1, conv2])
            
            # Create turns
            for conv in [conv1, conv2]:
                for i in range(conv.total_turns):
                    turn = Turn(
                        conversation_id=conv.id,
                        turn_number=i + 1,
                        speaker="test-model",
                        message="Test message"
                    )
                    db.session.add(turn)
            
            db.session.commit()
            
            stats = get_database_stats()
            
            assert stats['total_conversations'] == 2
            assert stats['completed_conversations'] == 1
            assert stats['active_conversations'] == 1
            assert stats['total_turns'] == 8  # 3 + 5
            assert stats['avg_turns_per_conversation'] == 4.0  # (3 + 5) / 2
            assert stats['avg_cost_per_conversation'] == 0.02  # (0.015 + 0.025) / 2
            assert stats['templates_used'] == 2  # debate, creative
            assert stats['models_used'] == 3  # model1, model2, model3
    
    def test_get_database_stats_empty(self, app):
        """Test getting database statistics with empty database."""
        with app.app_context():
            stats = get_database_stats()
            
            assert stats['total_conversations'] == 0
            assert stats['completed_conversations'] == 0
            assert stats['active_conversations'] == 0
            assert stats['total_turns'] == 0
            assert stats['avg_turns_per_conversation'] == 0
            assert stats['avg_cost_per_conversation'] == 0
            assert stats['templates_used'] == 0
            assert stats['models_used'] == 0


class TestDatabaseInitialization:
    """Test database initialization functions."""
    
    def test_init_database(self):
        """Test database initialization."""
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        # Should not raise any errors
        init_database(app)
        
        # Check that db is initialized
        assert db.app == app
    
    def test_create_tables(self):
        """Test table creation."""
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        with app.app_context():
            init_database(app)
            create_tables(app)
            
            # Check that we can create a conversation (tables exist)
            conv = Conversation(initial_prompt="Test")
            db.session.add(conv)
            db.session.commit()
            
            assert conv.id is not None