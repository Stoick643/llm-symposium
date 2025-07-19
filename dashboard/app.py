"""
Flask Dashboard for LLM Conversations

Basic Flask application for viewing and managing LLM conversations.
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from datetime import datetime
import tempfile

# Import our database models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from database_models import (
    db, init_database, create_tables,
    Conversation, Turn, QualityMetric,
    get_all_conversations, get_conversation_by_id,
    get_conversations_by_template, search_conversations,
    get_conversation_turns, get_database_stats
)
from tts_service import get_tts_service

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configuration
    import secrets
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_urlsafe(32))
    # Use absolute path for database
    project_root = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(project_root, 'instance', 'conversations.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{db_path}')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    init_database(app)
    
    # Initialize SocketIO with appropriate CORS settings
    cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000,http://127.0.0.1:3000').split(',')
    socketio = SocketIO(app, cors_allowed_origins=cors_origins)
    
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
        try:
            from template_manager import TemplateManager
            template_manager = TemplateManager()
            return render_template('templates.html', 
                                 templates=template_manager.templates)
        except Exception as e:
            return render_template('error.html', error=f"Template manager error: {str(e)}"), 500
    
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
    
    @socketio.on('join_live_conversation')
    def handle_join_live_conversation(data):
        """Join a live conversation room for real-time updates."""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            from flask_socketio import join_room
            join_room(f'live_{conversation_id}')
            
            # Update viewer count
            if conversation_id in live_conversations:
                live_conversations[conversation_id]['viewers'] += 1
                
                # Emit viewer count update to all viewers
                socketio.emit('viewer_count_update', {
                    'conversation_id': conversation_id,
                    'viewers': live_conversations[conversation_id]['viewers']
                }, room=f'live_{conversation_id}')
            
            emit('status', {'message': f'Joined live conversation {conversation_id}'})
    
    @socketio.on('leave_live_conversation')
    def handle_leave_live_conversation(data):
        """Leave a live conversation room."""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            from flask_socketio import leave_room
            leave_room(f'live_{conversation_id}')
            
            # Update viewer count
            if conversation_id in live_conversations:
                live_conversations[conversation_id]['viewers'] = max(0, 
                    live_conversations[conversation_id]['viewers'] - 1)
                
                # Emit viewer count update to all viewers
                socketio.emit('viewer_count_update', {
                    'conversation_id': conversation_id,
                    'viewers': live_conversations[conversation_id]['viewers']
                }, room=f'live_{conversation_id}')
            
            emit('status', {'message': f'Left live conversation {conversation_id}'})
    
    # TTS API endpoints
    @app.route('/api/generate-tts/<int:turn_id>', methods=['POST'])
    def generate_tts_for_turn(turn_id):
        """Generate TTS audio for a specific conversation turn."""
        try:
            # Get the turn from database
            turn = db.session.get(Turn, turn_id)
            if not turn:
                return jsonify({'error': 'Turn not found'}), 404
            
            # Get speaker from request or use turn's speaker
            data = request.get_json() or {}
            speaker = data.get('speaker', turn.speaker)
            
            # Initialize TTS service
            tts_service = get_tts_service()
            
            # Generate audio
            audio_file_path = tts_service.generate_speech_for_conversation_turn(
                text=turn.message,
                speaker_model=speaker
            )
            
            # Extract filename from path
            audio_filename = os.path.basename(audio_file_path)
            
            return jsonify({
                'success': True,
                'audio_filename': audio_filename,
                'turn_id': turn_id
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/audio/<filename>')
    def serve_audio(filename):
        """Serve generated audio files."""
        try:
            from flask import send_file
            from pathlib import Path
            
            # Try to get audio directory from TTS service, fallback to default
            try:
                tts_service = get_tts_service()
                audio_dir = tts_service.audio_dir
                print(f"TTS service audio directory: {audio_dir}")  # Debug logging
                
                # Ensure we have an absolute path
                if not audio_dir.is_absolute():
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    audio_dir = Path(project_root) / audio_dir
                    print(f"Converted to absolute path: {audio_dir}")  # Debug logging
                    
            except Exception as e:
                print(f"TTS service failed, using fallback: {e}")  # Debug logging
                # Fallback to default audio cache directory at project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                audio_dir = Path(project_root) / 'audio_cache'
                print(f"Fallback audio directory: {audio_dir}")  # Debug logging
            
            audio_path = audio_dir / filename
            print(f"Looking for audio file at: {audio_path}")  # Debug logging
            
            if not audio_path.exists():
                print(f"Audio file not found at: {audio_path}")  # Debug logging
                return jsonify({'error': f'Audio file not found: {filename}'}), 404
            
            return send_file(
                str(audio_path),
                mimetype='audio/mpeg',
                as_attachment=False
            )
            
        except Exception as e:
            import traceback
            print(f"Audio serving error: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Audio serving failed: {str(e)}'}), 500
    
    # Live conversation API endpoints
    live_conversations = {}  # Store active live conversations
    
    @app.route('/api/conversations/live', methods=['POST'])
    def start_live_conversation():
        """Start a new live conversation."""
        try:
            data = request.get_json() or {}
            template = data.get('template', 'debate')
            models = data.get('models', ['claude-3-sonnet-20240229', 'gpt-4'])
            topic = data.get('topic', 'General AI Discussion')
            
            # Create a unique conversation ID
            import uuid
            conversation_id = str(uuid.uuid4())
            
            # Store conversation metadata
            live_conversations[conversation_id] = {
                'id': conversation_id,
                'template': template,
                'models': models,
                'topic': topic,
                'status': 'ready',
                'viewers': 0,
                'created_at': datetime.utcnow().isoformat(),
                'current_turn': 0,
                'turns': []
            }
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'conversation': live_conversations[conversation_id]
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/conversations/live/<conversation_id>/start', methods=['POST'])
    def start_live_conversation_stream(conversation_id):
        """Start streaming a live conversation."""
        try:
            if conversation_id not in live_conversations:
                return jsonify({'error': 'Conversation not found'}), 404
            
            conversation = live_conversations[conversation_id]
            if conversation['status'] != 'ready':
                return jsonify({'error': 'Conversation not ready to start'}), 400
            
            # Update status to running
            conversation['status'] = 'running'
            
            # Emit to all viewers
            socketio.emit('conversation_started', {
                'conversation_id': conversation_id,
                'conversation': conversation
            }, room=f'live_{conversation_id}')
            
            # Start conversation in background thread
            import threading
            thread = threading.Thread(
                target=_run_live_conversation, 
                args=(conversation_id, socketio)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'status': 'running'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/conversations/live/<conversation_id>')
    def get_live_conversation(conversation_id):
        """Get live conversation status and data."""
        if conversation_id not in live_conversations:
            return jsonify({'error': 'Conversation not found'}), 404
        
        return jsonify({
            'conversation': live_conversations[conversation_id]
        })
    
    @app.route('/api/conversations/live')
    def list_live_conversations():
        """List all active live conversations."""
        return jsonify({
            'conversations': list(live_conversations.values())
        })
    
    @app.route('/api/conversations/live/<conversation_id>/stop', methods=['POST'])
    def stop_live_conversation(conversation_id):
        """Stop a live conversation."""
        try:
            if conversation_id not in live_conversations:
                return jsonify({'error': 'Conversation not found'}), 404
            
            conversation = live_conversations[conversation_id]
            conversation['status'] = 'stopped'
            
            # Emit to all viewers
            socketio.emit('conversation_stopped', {
                'conversation_id': conversation_id
            }, room=f'live_{conversation_id}')
            
            return jsonify({
                'success': True,
                'conversation_id': conversation_id,
                'status': 'stopped'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/live')
    def live_conversations_page():
        """Live conversations dashboard page."""
        return render_template('live.html')
    
    @app.route('/live/<conversation_id>')
    def live_conversation_viewer(conversation_id):
        """Live conversation viewer page."""
        if conversation_id not in live_conversations:
            return "Live conversation not found", 404
        
        conversation = live_conversations[conversation_id]
        return render_template('live_viewer.html', 
                             conversation=conversation,
                             conversation_id=conversation_id)
    
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
    
    def _run_live_conversation(conversation_id, socketio_instance):
        """Run a live conversation in the background."""
        try:
            if conversation_id not in live_conversations:
                return
            
            conversation = live_conversations[conversation_id]
            
            # Import conversation manager and config
            from conversation_manager import LLMConversation
            from config_manager import ConversationConfig
            from models import ConversationTurn
            from database_service import save_turn
            import os
            import time as time_module
            
            # Set up conversation configuration
            config = ConversationConfig(
                models=conversation['models'],
                template=conversation['template'],
                ai_aware_mode=True,
                mode='full',
                enable_quality_metrics=False
            )
            
            # Get API keys from environment
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
            moonshot_api_key = os.environ.get('MOONSHOT_API_KEY')
            
            if not api_key:
                raise Exception("ANTHROPIC_API_KEY environment variable not set")
            
            # Create conversation manager
            manager = LLMConversation(
                config=config,
                api_key=api_key,
                openai_api_key=openai_api_key,
                deepseek_api_key=deepseek_api_key,
                moonshot_api_key=moonshot_api_key,
                save_to_db=True
            )
            
            # Initialize database conversation
            initial_prompt = f"Let's have a {conversation['template']} about: {conversation['topic']}"
            manager._initialize_database_conversation(initial_prompt)
            
            # Start conversation loop
            turn_count = 0
            max_turns = 10  # Limit for live conversations
            current_prompt = initial_prompt
            
            while (conversation['status'] == 'running' and 
                   turn_count < max_turns and 
                   conversation_id in live_conversations):
                
                try:
                    # Get current model for this turn
                    current_model = manager.models[manager.current_model_index]
                    
                    # Get conversation context for current model
                    context = manager._build_context_for_model(current_model)
                    
                    # Get response from current model
                    response, input_tokens, output_tokens, response_time = manager._call_model(
                        current_model, current_prompt, context
                    )
                    
                    # Check for error responses
                    if response.startswith("Error:"):
                        print(f"Model error on turn {turn_count + 1}: {response}")
                        break
                    
                    # Update metrics
                    manager.total_input_tokens += input_tokens
                    manager.total_output_tokens += output_tokens
                    manager.response_times.append(response_time)
                    
                    # Store the turn
                    conversation_turn = ConversationTurn(
                        speaker=current_model,
                        message=response,
                        timestamp=time_module.time()
                    )
                    manager.conversation_history.append(conversation_turn)
                    
                    # Save turn to database if enabled
                    if manager.save_to_db and manager.conversation_id:
                        save_turn(
                            conversation_id=manager.conversation_id,
                            turn_number=turn_count + 1,
                            speaker=current_model,
                            message=response,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            response_time=response_time,
                            context_size=len(context)
                        )
                    
                    turn_count += 1
                    
                    # Update live conversation data
                    turn_data = {
                        'turn_number': turn_count,
                        'speaker': current_model,
                        'message': response,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    conversation['turns'].append(turn_data)
                    conversation['current_turn'] = turn_count
                    
                    # Emit the new turn to all viewers
                    socketio_instance.emit('new_turn', {
                        'conversation_id': conversation_id,
                        'turn': turn_data
                    }, room=f'live_{conversation_id}')
                    
                    # Switch to next model for next turn
                    manager.current_model_index = (manager.current_model_index + 1) % len(manager.models)
                    
                    # Update prompt for next turn (use the response as context)
                    current_prompt = response
                    
                    # Small delay between turns for better UX
                    time_module.sleep(2)
                    
                except Exception as e:
                    print(f"Error in conversation turn: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            # Mark conversation as completed
            conversation['status'] = 'completed'
            
            # Emit completion event
            socketio_instance.emit('conversation_completed', {
                'conversation_id': conversation_id,
                'total_turns': turn_count
            }, room=f'live_{conversation_id}')
            
        except Exception as e:
            print(f"Error running live conversation {conversation_id}: {e}")
            if conversation_id in live_conversations:
                live_conversations[conversation_id]['status'] = 'error'
                socketio_instance.emit('conversation_error', {
                    'conversation_id': conversation_id,
                    'error': str(e)
                }, room=f'live_{conversation_id}')
    
    return app, socketio


# Create the app
app, socketio = create_app()

if __name__ == '__main__':
    # Create tables on startup
    create_tables(app)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    is_development = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"Starting Flask dashboard on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Environment: {'development' if is_development else 'production'}")
    
    # Only allow unsafe Werkzeug in development
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, 
                allow_unsafe_werkzeug=is_development)