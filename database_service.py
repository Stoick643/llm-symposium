"""
Database Service Layer

Simple functions for saving conversations to the database.
Handles the database operations separately from conversation logic.
"""

from flask import Flask
from database_models import db, Conversation, Turn
from datetime import datetime
from typing import List, Optional
import json


class DatabaseService:
    """Service for managing conversation database operations."""
    
    def __init__(self, database_url="sqlite:///conversations.db"):
        """Initialize database service."""
        self.app = None
        self.database_url = database_url
        self._init_app()
    
    def _init_app(self):
        """Initialize Flask app for database context."""
        self.app = Flask(__name__)
        self.app.config['SQLALCHEMY_DATABASE_URI'] = self.database_url
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        
        db.init_app(self.app)
        
        # Create tables if they don't exist
        with self.app.app_context():
            db.create_all()
    
    def create_conversation(self, models: List[str], initial_prompt: str, 
                          mode: str = "full", window_size: int = 10,
                          template: Optional[str] = None, 
                          ai_aware_mode: bool = False) -> Optional[int]:
        """
        Create a new conversation in the database.
        
        Returns:
            Conversation ID if successful, None if failed
        """
        try:
            with self.app.app_context():
                conversation = Conversation(
                    initial_prompt=initial_prompt,
                    mode=mode,
                    window_size=window_size,
                    template=template,
                    ai_aware_mode=ai_aware_mode,
                    status="active"
                )
                
                # Set models using our helper method
                conversation.set_models(models)
                
                db.session.add(conversation)
                db.session.commit()
                
                print(f"💾 Created conversation {conversation.id} in database")
                return conversation.id
                
        except Exception as e:
            print(f"❌ Failed to create conversation: {e}")
            return None
    
    def save_turn(self, conversation_id: int, turn_number: int, 
                  speaker: str, message: str, input_tokens: int = 0,
                  output_tokens: int = 0, response_time: float = 0.0,
                  context_size: int = 0) -> bool:
        """
        Save a conversation turn to the database.
        
        Returns:
            True if successful, False if failed
        """
        try:
            with self.app.app_context():
                turn = Turn(
                    conversation_id=conversation_id,
                    turn_number=turn_number,
                    speaker=speaker,
                    message=message,
                    timestamp=datetime.utcnow(),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    response_time=response_time,
                    context_size=context_size
                )
                
                db.session.add(turn)
                db.session.commit()
                
                print(f"💾 Saved turn {turn_number} for conversation {conversation_id}")
                return True
                
        except Exception as e:
            print(f"❌ Failed to save turn: {e}")
            return False
    
    def update_conversation_metrics(self, conversation_id: int, 
                                  total_turns: int, total_cost: float,
                                  total_input_tokens: int, total_output_tokens: int,
                                  avg_response_time: float, status: str = "completed") -> bool:
        """
        Update conversation metrics when complete.
        
        Returns:
            True if successful, False if failed
        """
        try:
            with self.app.app_context():
                conversation = Conversation.query.get(conversation_id)
                if not conversation:
                    print(f"❌ Conversation {conversation_id} not found")
                    return False
                
                conversation.total_turns = total_turns
                conversation.total_cost = total_cost
                conversation.total_input_tokens = total_input_tokens
                conversation.total_output_tokens = total_output_tokens
                conversation.avg_response_time = avg_response_time
                conversation.status = status
                
                db.session.commit()
                
                print(f"💾 Updated metrics for conversation {conversation_id}")
                return True
                
        except Exception as e:
            print(f"❌ Failed to update conversation metrics: {e}")
            return False
    
    def get_conversation(self, conversation_id: int) -> Optional[dict]:
        """Get conversation by ID."""
        try:
            with self.app.app_context():
                conversation = Conversation.query.get(conversation_id)
                if conversation:
                    return conversation.to_dict()
                return None
        except Exception as e:
            print(f"❌ Failed to get conversation: {e}")
            return None
    
    def get_recent_conversations(self, limit: int = 10) -> List[dict]:
        """Get recent conversations."""
        try:
            with self.app.app_context():
                conversations = Conversation.query.order_by(
                    Conversation.created_at.desc()
                ).limit(limit).all()
                
                return [conv.to_dict() for conv in conversations]
        except Exception as e:
            print(f"❌ Failed to get recent conversations: {e}")
            return []


# Global database service instance
db_service = DatabaseService()


# Simple functions for easy import
def create_conversation(models: List[str], initial_prompt: str, **kwargs) -> Optional[int]:
    """Create a new conversation. Returns conversation ID."""
    return db_service.create_conversation(models, initial_prompt, **kwargs)


def save_turn(conversation_id: int, turn_number: int, speaker: str, 
              message: str, **kwargs) -> bool:
    """Save a conversation turn."""
    return db_service.save_turn(conversation_id, turn_number, speaker, message, **kwargs)


def update_metrics(conversation_id: int, **kwargs) -> bool:
    """Update conversation metrics."""
    return db_service.update_conversation_metrics(conversation_id, **kwargs)


def get_conversation(conversation_id: int) -> Optional[dict]:
    """Get conversation by ID."""
    return db_service.get_conversation(conversation_id)


def get_recent_conversations(limit: int = 10) -> List[dict]:
    """Get recent conversations."""
    return db_service.get_recent_conversations(limit)