"""
Flask Dashboard for LLM Conversations

Basic Flask application for viewing and managing LLM conversations.
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from datetime import datetime

# Import our database models
import sys
sys.path.append('..')
from database_models import (
    db, init_database, create_tables,
    Conversation, Turn, QualityMetric,
    get_all_conversations, get_conversation_by_id,
    get_conversations_by_template, search_conversations,
    get_conversation_turns, get_database_stats
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///conversations.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    init_database(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Routes
    @app.route('/')
    def index():
        """Main dashboard page."""
        page = request.args.get('page', 1, type=int)
        conversations = get_all_conversations(page=page, per_page=10)
        stats = get_database_stats()
        
        return render_template('index.html', 
                             conversations=conversations, 
                             stats=stats)
    
    @app.route('/conversation/<int:conversation_id>')
    def conversation_detail(conversation_id):
        """Detailed view of a specific conversation."""
        conversation = get_conversation_by_id(conversation_id)
        if not conversation:
            return "Conversation not found", 404
        
        turns = get_conversation_turns(conversation_id)
        return render_template('conversation.html', 
                             conversation=conversation, 
                             turns=turns)
    
    @app.route('/api/conversations')
    def api_conversations():
        """API endpoint for conversations."""
        page = request.args.get('page', 1, type=int)
        template = request.args.get('template')
        search_query = request.args.get('q')
        
        if search_query:
            conversations = search_conversations(search_query, page=page, per_page=10)
        elif template:
            conversations = get_conversations_by_template(template, page=page, per_page=10)
        else:
            conversations = get_all_conversations(page=page, per_page=10)
        
        return jsonify({
            'conversations': [conv.to_dict() for conv in conversations.items],
            'pagination': {
                'page': conversations.page,
                'pages': conversations.pages,
                'per_page': conversations.per_page,
                'total': conversations.total,
                'has_next': conversations.has_next,
                'has_prev': conversations.has_prev
            }
        })
    
    @app.route('/api/conversation/<int:conversation_id>')
    def api_conversation_detail(conversation_id):
        """API endpoint for conversation details."""
        conversation = get_conversation_by_id(conversation_id)
        if not conversation:
            return jsonify({'error': 'Conversation not found'}), 404
        
        turns = get_conversation_turns(conversation_id)
        return jsonify({
            'conversation': conversation.to_dict(),
            'turns': [turn.to_dict() for turn in turns]
        })
    
    @app.route('/api/stats')
    def api_stats():
        """API endpoint for database statistics."""
        stats = get_database_stats()
        return jsonify(stats)
    
    @app.route('/templates')
    def templates():
        """Templates management page."""
        from template_manager import TemplateManager
        template_manager = TemplateManager()
        return render_template('templates.html', 
                             templates=template_manager.templates)
    
    # WebSocket events
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        print('Client connected')
        emit('status', {'message': 'Connected to dashboard'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        print('Client disconnected')
    
    @socketio.on('join_conversation')
    def handle_join_conversation(data):
        """Join a conversation room for real-time updates."""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            from flask_socketio import join_room
            join_room(f'conversation_{conversation_id}')
            emit('status', {'message': f'Joined conversation {conversation_id}'})
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', error="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', error="Internal server error"), 500
    
    # CLI commands
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        create_tables(app)
        print("Database initialized successfully")
    
    @app.cli.command()
    def stats():
        """Show database statistics."""
        stats = get_database_stats()
        print("Database Statistics:")
        print(f"Total Conversations: {stats['total_conversations']}")
        print(f"Completed Conversations: {stats['completed_conversations']}")
        print(f"Active Conversations: {stats['active_conversations']}")
        print(f"Total Turns: {stats['total_turns']}")
        print(f"Average Turns per Conversation: {stats['avg_turns_per_conversation']:.1f}")
        print(f"Average Cost per Conversation: ${stats['avg_cost_per_conversation']:.4f}")
        print(f"Templates Used: {stats['templates_used']}")
        print(f"Models Used: {stats['models_used']}")
    
    return app, socketio


# Create the app
app, socketio = create_app()

if __name__ == '__main__':
    # Create tables on startup
    create_tables(app)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Flask dashboard on port {port}")
    print(f"Debug mode: {debug}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=debug)