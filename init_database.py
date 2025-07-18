#!/usr/bin/env python3
"""
Database Initialization Script

Creates SQLite database and tables for LLM conversations.
"""

import os
import sys
from flask import Flask
from database_models import db, create_tables, Conversation, Turn, QualityMetric
from datetime import datetime
import json

def create_app(database_url="sqlite:///conversations.db"):
    """Create Flask app with database configuration."""
    app = Flask(__name__)
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    db.init_app(app)
    
    return app

def init_database(database_url="sqlite:///conversations.db"):
    """Initialize the database with tables."""
    print(f"🔧 Initializing database: {database_url}")
    
    app = create_app(database_url)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Check if we have any conversations
        conversation_count = Conversation.query.count()
        turn_count = Turn.query.count()
        
        print(f"📊 Current database stats:")
        print(f"  - Conversations: {conversation_count}")
        print(f"  - Turns: {turn_count}")
        
        return app

def add_sample_data():
    """Add sample conversation data for testing."""
    print("📝 Adding sample conversation data...")
    
    # Create sample conversation
    sample_conversation = Conversation(
        initial_prompt="Hello, let's have a quick test conversation about AI.",
        mode="full",
        window_size=10,
        template=None,
        ai_aware_mode=False,
        status="completed",
        total_turns=2,
        total_cost=0.0012,
        total_input_tokens=50,
        total_output_tokens=80,
        avg_response_time=2.5
    )
    
    # Set models using our helper method
    sample_conversation.set_models(["deepseek-chat", "kimi-k2-0711-preview"])
    
    db.session.add(sample_conversation)
    db.session.flush()  # Get the ID
    
    # Add sample turns
    turn1 = Turn(
        conversation_id=sample_conversation.id,
        turn_number=1,
        speaker="deepseek-chat",
        message="Hello! I'm DeepSeek, an AI assistant. I'm here to help with various tasks and have conversations. What would you like to talk about?",
        timestamp=datetime.utcnow(),
        input_tokens=25,
        output_tokens=35,
        response_time=2.1,
        context_size=1
    )
    
    turn2 = Turn(
        conversation_id=sample_conversation.id,
        turn_number=2,
        speaker="kimi-k2-0711-preview",
        message="Hi there! I'm Kimi, your AI assistant and friend. It's nice to meet another AI! I enjoy having conversations about technology, creativity, and helping people with their questions.",
        timestamp=datetime.utcnow(),
        input_tokens=25,
        output_tokens=45,
        response_time=2.9,
        context_size=2
    )
    
    db.session.add(turn1)
    db.session.add(turn2)
    db.session.commit()
    
    print("✅ Sample data added successfully")
    print(f"  - Conversation ID: {sample_conversation.id}")
    print(f"  - Models: {sample_conversation.get_models()}")
    print(f"  - Turns: {len([turn1, turn2])}")

def show_database_info():
    """Display database information."""
    print("\n📊 Database Information:")
    
    conversations = Conversation.query.all()
    print(f"Total conversations: {len(conversations)}")
    
    for conv in conversations:
        print(f"\n🗣️ {conv}")
        print(f"   Created: {conv.created_at}")
        print(f"   Models: {conv.get_models()}")
        print(f"   Mode: {conv.mode}")
        print(f"   Turns: {conv.total_turns}")
        print(f"   Cost: ${conv.total_cost}")
        print(f"   Status: {conv.status}")
        
        # Show first few turns
        turns = Turn.query.filter_by(conversation_id=conv.id).limit(3).all()
        for turn in turns:
            preview = turn.message[:50] + "..." if len(turn.message) > 50 else turn.message
            print(f"   Turn {turn.turn_number} ({turn.speaker}): {preview}")

def reset_database():
    """Reset the database (drop and recreate all tables)."""
    print("⚠️  Resetting database - this will delete all data!")
    response = input("Are you sure? (yes/no): ")
    
    if response.lower() != 'yes':
        print("❌ Database reset cancelled")
        return
    
    # Drop all tables
    db.drop_all()
    print("🗑️  All tables dropped")
    
    # Recreate tables
    db.create_all()
    print("✅ Tables recreated")

def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python3 init_database.py <command>")
        print("Commands:")
        print("  init         - Initialize database with tables")
        print("  sample       - Add sample conversation data")
        print("  info         - Show database information")
        print("  reset        - Reset database (delete all data)")
        return
    
    command = sys.argv[1]
    
    # Initialize app
    app = init_database()
    
    with app.app_context():
        if command == "init":
            print("✅ Database initialization complete!")
            
        elif command == "sample":
            add_sample_data()
            
        elif command == "info":
            show_database_info()
            
        elif command == "reset":
            reset_database()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: init, sample, info, reset")

if __name__ == "__main__":
    main()