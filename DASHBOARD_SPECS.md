# Flask Dashboard Specifications

This document outlines the architecture and implementation specifications for the Flask web dashboard that provides real-time conversation viewing, historical replay, and AI-generated analysis.

## Overview

The Flask dashboard transforms the CLI-based LLM conversation tool into a comprehensive web application with:
- Real-time conversation streaming via WebSockets
- Historical conversation replay with playback controls
- AI-generated summaries and keywords for searchability
- Text-to-speech integration
- Mobile-responsive design

## Database Architecture

### Schema Design

#### Conversations Table
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Configuration
    model_1 TEXT NOT NULL,
    model_2 TEXT NOT NULL,
    mode TEXT NOT NULL,                    -- 'full', 'sliding', 'cache', 'sliding_cache'
    window_size INTEGER,
    template TEXT,
    initial_prompt TEXT NOT NULL,
    ai_aware_mode BOOLEAN DEFAULT FALSE,
    
    -- Metrics
    total_turns INTEGER DEFAULT 0,
    total_cost DECIMAL(10,6) DEFAULT 0.0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    avg_response_time DECIMAL(5,3),
    
    -- Status
    status TEXT DEFAULT 'active',          -- 'active', 'completed', 'error'
    
    -- AI-generated content
    summary TEXT,
    keywords TEXT,                         -- JSON array of keywords
    topic_category TEXT,
    summary_generated_at TIMESTAMP,
    keywords_generated_at TIMESTAMP,
    summary_model TEXT,
    keywords_model TEXT
);
```

#### Turns Table
```sql
CREATE TABLE turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    turn_number INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Metrics
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    response_time DECIMAL(5,3),
    context_size INTEGER,
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

#### Quality Metrics Table (Optional)
```sql
CREATE TABLE quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    turn_number INTEGER,
    coherence_score DECIMAL(5,3),
    topic_drift_score DECIMAL(5,3),
    repetition_rate DECIMAL(5,3),
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

### Indexes for Performance
```sql
CREATE INDEX idx_conversations_created_at ON conversations(created_at);
CREATE INDEX idx_conversations_status ON conversations(status);
CREATE INDEX idx_conversations_template ON conversations(template);
CREATE INDEX idx_turns_conversation_id ON turns(conversation_id);
CREATE INDEX idx_turns_timestamp ON turns(timestamp);
CREATE INDEX idx_quality_metrics_conversation_id ON quality_metrics(conversation_id);
```

## Flask Application Architecture

### Project Structure
```
dashboard/
├── app.py                 # Main Flask application
├── models.py              # SQLAlchemy models
├── database.py            # Database initialization and utilities
├── conversation_analyzer.py # AI summary/keyword generation
├── websocket_handlers.py  # WebSocket event handlers
├── api/
│   ├── __init__.py
│   ├── conversations.py   # Conversation API endpoints
│   └── search.py         # Search and filtering endpoints
├── templates/
│   ├── base.html         # Base template
│   ├── dashboard.html    # Main dashboard
│   ├── conversation.html # Individual conversation view
│   └── components/       # Reusable template components
├── static/
│   ├── css/
│   │   └── main.css     # Tailwind CSS build
│   ├── js/
│   │   ├── websocket.js # WebSocket client
│   │   ├── replay.js    # Historical replay functionality
│   │   └── tts.js       # Text-to-speech integration
│   └── images/
└── migrations/           # Database migration scripts
```

### Core Components

#### Flask Application (app.py)
```python
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import json
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Import models and handlers
from models import Conversation, Turn, QualityMetric
from websocket_handlers import register_websocket_handlers

register_websocket_handlers(socketio)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/conversation/<int:conversation_id>')
def conversation_view(conversation_id):
    conversation = Conversation.query.get_or_404(conversation_id)
    return render_template('conversation.html', conversation=conversation)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
```

#### SQLAlchemy Models (models.py)
```python
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()

class Conversation(db.Model):
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configuration
    model_1 = db.Column(db.String(100), nullable=False)
    model_2 = db.Column(db.String(100), nullable=False)
    mode = db.Column(db.String(20), nullable=False)
    window_size = db.Column(db.Integer)
    template = db.Column(db.String(50))
    initial_prompt = db.Column(db.Text, nullable=False)
    ai_aware_mode = db.Column(db.Boolean, default=False)
    
    # Metrics
    total_turns = db.Column(db.Integer, default=0)
    total_cost = db.Column(db.Numeric(10, 6), default=0.0)
    total_input_tokens = db.Column(db.Integer, default=0)
    total_output_tokens = db.Column(db.Integer, default=0)
    avg_response_time = db.Column(db.Numeric(5, 3))
    
    # Status
    status = db.Column(db.String(20), default='active')
    
    # AI-generated content
    summary = db.Column(db.Text)
    keywords = db.Column(db.Text)  # JSON array
    topic_category = db.Column(db.String(50))
    summary_generated_at = db.Column(db.DateTime)
    keywords_generated_at = db.Column(db.DateTime)
    summary_model = db.Column(db.String(50))
    keywords_model = db.Column(db.String(50))
    
    # Relationships
    turns = db.relationship('Turn', backref='conversation', lazy=True, cascade='all, delete-orphan')
    quality_metrics = db.relationship('QualityMetric', backref='conversation', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'model_1': self.model_1,
            'model_2': self.model_2,
            'mode': self.mode,
            'template': self.template,
            'initial_prompt': self.initial_prompt,
            'total_turns': self.total_turns,
            'total_cost': float(self.total_cost) if self.total_cost else 0.0,
            'status': self.status,
            'summary': self.summary,
            'keywords': json.loads(self.keywords) if self.keywords else [],
            'topic_category': self.topic_category
        }

class Turn(db.Model):
    __tablename__ = 'turns'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    turn_number = db.Column(db.Integer, nullable=False)
    speaker = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    
    # Metrics
    input_tokens = db.Column(db.Integer, default=0)
    output_tokens = db.Column(db.Integer, default=0)
    response_time = db.Column(db.Numeric(5, 3))
    context_size = db.Column(db.Integer)
    
    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'turn_number': self.turn_number,
            'speaker': self.speaker,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'response_time': float(self.response_time) if self.response_time else 0.0,
            'context_size': self.context_size
        }

class QualityMetric(db.Model):
    __tablename__ = 'quality_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    turn_number = db.Column(db.Integer)
    coherence_score = db.Column(db.Numeric(5, 3))
    topic_drift_score = db.Column(db.Numeric(5, 3))
    repetition_rate = db.Column(db.Numeric(5, 3))
```

## AI Generation System

### ConversationAnalyzer Service
```python
from openai import OpenAI
import json
import logging
from datetime import datetime

class ConversationAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
    def generate_summary(self, conversation_turns, max_chars=4000):
        """Generate AI-powered conversation summary"""
        try:
            conversation_text = self._prepare_conversation_text(conversation_turns, max_chars)
            
            prompt = f"""
            Analyze this AI-to-AI conversation and provide a concise 2-3 sentence summary.

            Focus on:
            - Main topics and themes discussed
            - Key conclusions or insights reached
            - Type of interaction (debate, collaboration, technical discussion, etc.)

            Conversation:
            {conversation_text}

            Summary:"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            self.logger.info(f"Generated summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            return None
    
    def extract_keywords(self, conversation_turns, max_chars=4000):
        """Extract AI-powered keywords from conversation"""
        try:
            conversation_text = self._prepare_conversation_text(conversation_turns, max_chars)
            
            prompt = f"""
            Extract 5-10 relevant keywords/tags from this AI-to-AI conversation.

            Focus on:
            - Main topics and concepts discussed
            - Technologies or methodologies mentioned
            - Themes and subject areas
            - Specific terms that would help someone search for this conversation

            Return only the keywords as a JSON array (e.g., ["keyword1", "keyword2", "keyword3"]).

            Conversation:
            {conversation_text}

            Keywords:"""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2
            )
            
            keywords_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                keywords = json.loads(keywords_text)
                if isinstance(keywords, list):
                    # Clean and validate keywords
                    keywords = [k.strip() for k in keywords if k.strip()][:10]  # Max 10 keywords
                    self.logger.info(f"Extracted keywords: {keywords}")
                    return keywords
                else:
                    self.logger.warning(f"Invalid keywords format: {keywords_text}")
                    return []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse keywords JSON: {keywords_text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to extract keywords: {str(e)}")
            return []
    
    def analyze_conversation(self, conversation_id):
        """Complete analysis of a conversation"""
        from models import Conversation, Turn, db
        
        conversation = Conversation.query.get(conversation_id)
        if not conversation:
            self.logger.error(f"Conversation {conversation_id} not found")
            return False
            
        turns = Turn.query.filter_by(conversation_id=conversation_id).order_by(Turn.turn_number).all()
        if not turns:
            self.logger.warning(f"No turns found for conversation {conversation_id}")
            return False
        
        # Generate summary and keywords
        summary = self.generate_summary(turns)
        keywords = self.extract_keywords(turns)
        
        # Update conversation record
        if summary:
            conversation.summary = summary
            conversation.summary_generated_at = datetime.utcnow()
            conversation.summary_model = "gpt-3.5-turbo"
        
        if keywords:
            conversation.keywords = json.dumps(keywords)
            conversation.keywords_generated_at = datetime.utcnow()
            conversation.keywords_model = "gpt-3.5-turbo"
        
        # Simple topic categorization based on template or keywords
        if conversation.template:
            conversation.topic_category = conversation.template
        elif keywords:
            conversation.topic_category = self._categorize_by_keywords(keywords)
        
        try:
            db.session.commit()
            self.logger.info(f"Successfully analyzed conversation {conversation_id}")
            return True
        except Exception as e:
            db.session.rollback()
            self.logger.error(f"Failed to save analysis for conversation {conversation_id}: {str(e)}")
            return False
    
    def _prepare_conversation_text(self, conversation_turns, max_chars=4000):
        """Prepare conversation text for AI analysis"""
        text_parts = []
        
        for turn in conversation_turns:
            # Use simple Model A/B labels instead of full model names
            speaker_label = "Model A" if turn.turn_number % 2 == 1 else "Model B"
            text_parts.append(f"{speaker_label}: {turn.message}")
        
        full_text = "\n\n".join(text_parts)
        
        # Truncate if too long
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n[Conversation truncated for analysis]"
        
        return full_text
    
    def _categorize_by_keywords(self, keywords):
        """Simple keyword-based categorization"""
        tech_keywords = ['algorithm', 'programming', 'software', 'code', 'technical', 'system', 'architecture']
        philosophy_keywords = ['ethics', 'philosophy', 'consciousness', 'meaning', 'morality', 'existence']
        science_keywords = ['research', 'experiment', 'hypothesis', 'scientific', 'analysis', 'data']
        creative_keywords = ['story', 'creative', 'art', 'design', 'imagination', 'narrative']
        
        keyword_lower = [k.lower() for k in keywords]
        
        if any(k in keyword_lower for k in tech_keywords):
            return 'technical'
        elif any(k in keyword_lower for k in philosophy_keywords):
            return 'philosophical'
        elif any(k in keyword_lower for k in science_keywords):
            return 'scientific'
        elif any(k in keyword_lower for k in creative_keywords):
            return 'creative'
        else:
            return 'general'
```

## WebSocket Real-time Updates

### WebSocket Event Handlers
```python
from flask_socketio import emit, join_room, leave_room
from models import Conversation, Turn, db
import json

def register_websocket_handlers(socketio):
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('status', {'message': 'Connected to dashboard'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
    
    @socketio.on('join_conversation')
    def handle_join_conversation(data):
        """Join a conversation room for real-time updates"""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            join_room(f'conversation_{conversation_id}')
            emit('status', {'message': f'Joined conversation {conversation_id}'})
    
    @socketio.on('leave_conversation')
    def handle_leave_conversation(data):
        """Leave a conversation room"""
        conversation_id = data.get('conversation_id')
        if conversation_id:
            leave_room(f'conversation_{conversation_id}')
    
    @socketio.on('start_replay')
    def handle_start_replay(data):
        """Start replaying a historical conversation"""
        conversation_id = data.get('conversation_id')
        speed = data.get('speed', 1.0)
        
        if not conversation_id:
            emit('error', {'message': 'No conversation ID provided'})
            return
        
        # Start replay in background thread
        socketio.start_background_task(replay_conversation, conversation_id, speed)
    
    @socketio.on('pause_replay')
    def handle_pause_replay(data):
        """Pause ongoing replay"""
        # Implementation depends on replay state management
        emit('replay_paused', {'message': 'Replay paused'})
    
    def replay_conversation(conversation_id, speed=1.0):
        """Background task to replay conversation turns"""
        import time
        from datetime import datetime
        
        turns = Turn.query.filter_by(conversation_id=conversation_id).order_by(Turn.turn_number).all()
        
        if not turns:
            socketio.emit('error', {'message': 'No turns found for conversation'})
            return
        
        # Emit replay start
        socketio.emit('replay_started', {
            'conversation_id': conversation_id,
            'total_turns': len(turns),
            'speed': speed
        }, room=f'conversation_{conversation_id}')
        
        previous_timestamp = None
        
        for turn in turns:
            # Calculate delay based on original timing
            if previous_timestamp:
                original_delay = (turn.timestamp - previous_timestamp).total_seconds()
                actual_delay = max(0.1, original_delay / speed)  # Minimum 0.1s delay
            else:
                actual_delay = 0.5 / speed  # Initial delay
            
            time.sleep(actual_delay)
            
            # Emit turn data
            socketio.emit('replay_turn', {
                'turn': turn.to_dict(),
                'is_replay': True
            }, room=f'conversation_{conversation_id}')
            
            previous_timestamp = turn.timestamp
        
        # Emit replay complete
        socketio.emit('replay_completed', {
            'conversation_id': conversation_id
        }, room=f'conversation_{conversation_id}')

# Function to emit real-time conversation updates (called from CLI app)
def emit_conversation_update(conversation_id, turn_data):
    """Emit real-time conversation updates to dashboard"""
    socketio.emit('conversation_turn', {
        'conversation_id': conversation_id,
        'turn': turn_data,
        'is_replay': False
    }, room=f'conversation_{conversation_id}')

def emit_conversation_completed(conversation_id, metrics):
    """Emit conversation completion notification"""
    socketio.emit('conversation_completed', {
        'conversation_id': conversation_id,
        'metrics': metrics
    }, room=f'conversation_{conversation_id}')
```

## Frontend Implementation

### Main Dashboard Template (dashboard.html)
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Conversation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
</head>
<body class="bg-gray-100 min-h-screen">
    <div id="app" class="container mx-auto px-4 py-8">
        <!-- Header -->
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">LLM Conversation Dashboard</h1>
            <div class="flex flex-col md:flex-row gap-4 items-start md:items-center">
                <div class="flex-1">
                    <input type="search" id="searchInput" placeholder="Search conversations, summaries, or keywords..." 
                           class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                </div>
                <div class="flex gap-2">
                    <button class="filter-btn px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm" data-filter="all">All</button>
                    <button class="filter-btn px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm" data-filter="technical">Technical</button>
                    <button class="filter-btn px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm" data-filter="debate">Debate</button>
                    <button class="filter-btn px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm" data-filter="creative">Creative</button>
                </div>
            </div>
        </header>

        <!-- Live Conversations Section -->
        <section class="mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Live Conversations</h2>
            <div id="liveConversations" class="grid gap-4">
                <!-- Live conversation cards will be populated here -->
            </div>
        </section>

        <!-- Historical Conversations Section -->
        <section>
            <h2 class="text-xl font-semibold text-gray-800 mb-4">Historical Conversations</h2>
            <div id="conversationGrid" class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                <!-- Conversation cards will be populated here -->
            </div>
        </section>

        <!-- Conversation Modal -->
        <div id="conversationModal" class="fixed inset-0 bg-black bg-opacity-50 hidden z-50">
            <div class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden">
                    <div class="p-6 border-b border-gray-200">
                        <div class="flex items-center justify-between">
                            <h3 id="modalTitle" class="text-xl font-semibold"></h3>
                            <button id="closeModal" class="text-gray-400 hover:text-gray-600">
                                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                </svg>
                            </button>
                        </div>
                        
                        <!-- Replay Controls -->
                        <div id="replayControls" class="mt-4 flex items-center gap-4">
                            <button id="playPauseBtn" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                                Play
                            </button>
                            <select id="speedSelect" class="px-3 py-1 border border-gray-300 rounded">
                                <option value="0.5">0.5x</option>
                                <option value="1" selected>1x</option>
                                <option value="2">2x</option>
                                <option value="5">5x</option>
                            </select>
                            <span id="replayProgress" class="text-sm text-gray-600"></span>
                        </div>
                    </div>
                    
                    <div id="conversationContent" class="p-6 overflow-y-auto max-h-[60vh]">
                        <!-- Conversation turns will be populated here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- TTS Controls -->
    <div id="ttsControls" class="fixed bottom-4 right-4 bg-white rounded-lg shadow-lg p-4 hidden">
        <div class="flex items-center gap-2">
            <button id="ttsToggle" class="px-3 py-1 bg-green-500 text-white rounded text-sm">
                🔊 TTS On
            </button>
            <select id="voiceSelect" class="px-2 py-1 border border-gray-300 rounded text-sm">
                <!-- Voice options will be populated by JavaScript -->
            </select>
            <input type="range" id="speechRate" min="0.5" max="2" step="0.1" value="1" class="w-20">
        </div>
    </div>

    <script src="{{ url_for('static', filename='js/websocket.js') }}"></script>
    <script src="{{ url_for('static', filename='js/replay.js') }}"></script>
    <script src="{{ url_for('static', filename='js/tts.js') }}"></script>
    <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
</body>
</html>
```

### WebSocket Client (static/js/websocket.js)
```javascript
class WebSocketClient {
    constructor() {
        this.socket = io();
        this.currentConversationId = null;
        this.isReplayActive = false;
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.showStatus('Connected to dashboard');
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.showStatus('Disconnected from dashboard');
        });

        this.socket.on('conversation_turn', (data) => {
            this.handleConversationTurn(data);
        });

        this.socket.on('conversation_completed', (data) => {
            this.handleConversationCompleted(data);
        });

        this.socket.on('replay_started', (data) => {
            this.handleReplayStarted(data);
        });

        this.socket.on('replay_turn', (data) => {
            this.handleReplayTurn(data);
        });

        this.socket.on('replay_completed', (data) => {
            this.handleReplayCompleted(data);
        });

        this.socket.on('error', (data) => {
            console.error('WebSocket error:', data);
            this.showStatus(`Error: ${data.message}`, 'error');
        });
    }

    joinConversation(conversationId) {
        this.currentConversationId = conversationId;
        this.socket.emit('join_conversation', { conversation_id: conversationId });
    }

    leaveConversation() {
        if (this.currentConversationId) {
            this.socket.emit('leave_conversation', { conversation_id: this.currentConversationId });
            this.currentConversationId = null;
        }
    }

    startReplay(conversationId, speed = 1.0) {
        this.isReplayActive = true;
        this.socket.emit('start_replay', { 
            conversation_id: conversationId, 
            speed: speed 
        });
    }

    pauseReplay() {
        this.socket.emit('pause_replay', {});
    }

    handleConversationTurn(data) {
        const { conversation_id, turn, is_replay } = data;
        
        if (is_replay) {
            this.addTurnToReplay(turn);
        } else {
            this.addTurnToLive(conversation_id, turn);
        }

        // Trigger TTS if enabled
        if (window.ttsManager && window.ttsManager.isEnabled()) {
            window.ttsManager.speakTurn(turn);
        }
    }

    handleConversationCompleted(data) {
        const { conversation_id, metrics } = data;
        this.showConversationCompleted(conversation_id, metrics);
    }

    handleReplayStarted(data) {
        const { conversation_id, total_turns, speed } = data;
        this.showReplayProgress(0, total_turns, speed);
        document.getElementById('playPauseBtn').textContent = 'Pause';
    }

    handleReplayTurn(data) {
        const { turn } = data;
        this.addTurnToReplay(turn);
        this.updateReplayProgress(turn.turn_number);
    }

    handleReplayCompleted(data) {
        this.isReplayActive = false;
        document.getElementById('playPauseBtn').textContent = 'Replay';
        this.showStatus('Replay completed');
    }

    addTurnToReplay(turn) {
        const container = document.getElementById('conversationContent');
        const turnElement = this.createTurnElement(turn, true);
        container.appendChild(turnElement);
        container.scrollTop = container.scrollHeight;
    }

    addTurnToLive(conversationId, turn) {
        // Update live conversation display
        const liveContainer = document.getElementById('liveConversations');
        // Implementation depends on live conversation UI structure
    }

    createTurnElement(turn, isReplay = false) {
        const div = document.createElement('div');
        div.className = `turn-message mb-4 p-4 rounded-lg ${isReplay ? 'bg-gray-50' : 'bg-white'} border-l-4 ${
            turn.turn_number % 2 === 1 ? 'border-blue-500' : 'border-green-500'
        }`;

        const speaker = turn.turn_number % 2 === 1 ? 'Model A' : 'Model B';
        const speakerColor = turn.turn_number % 2 === 1 ? 'text-blue-600' : 'text-green-600';

        div.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <span class="font-semibold ${speakerColor}">${speaker}</span>
                <span class="text-sm text-gray-500">${new Date(turn.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="message-content text-gray-800 whitespace-pre-wrap">${turn.message}</div>
            <div class="mt-2 text-xs text-gray-500">
                Tokens: ${turn.input_tokens} in, ${turn.output_tokens} out | 
                Response time: ${turn.response_time}s
            </div>
        `;

        return div;
    }

    showStatus(message, type = 'info') {
        // Show status message to user
        console.log(`[${type}] ${message}`);
    }

    showReplayProgress(current, total, speed) {
        const progressElement = document.getElementById('replayProgress');
        if (progressElement) {
            progressElement.textContent = `${current}/${total} turns (${speed}x speed)`;
        }
    }

    updateReplayProgress(currentTurn) {
        // Update progress indicator
        const progressElement = document.getElementById('replayProgress');
        if (progressElement) {
            const currentText = progressElement.textContent;
            const match = currentText.match(/(\d+)\/(\d+)/);
            if (match) {
                const total = match[2];
                const speed = currentText.match(/\(([\d.]+)x speed\)/)?.[1] || '1';
                progressElement.textContent = `${currentTurn}/${total} turns (${speed}x speed)`;
            }
        }
    }

    showConversationCompleted(conversationId, metrics) {
        this.showStatus(`Conversation ${conversationId} completed. Cost: $${metrics.total_cost}`);
    }
}

// Initialize WebSocket client
window.wsClient = new WebSocketClient();
```

### Text-to-Speech Integration (static/js/tts.js)
```javascript
class TTSManager {
    constructor() {
        this.enabled = false;
        this.voices = [];
        this.currentVoice = null;
        this.speechRate = 1.0;
        this.modelAVoice = null;
        this.modelBVoice = null;
        this.currentSpeech = null;
        
        this.initialize();
    }

    initialize() {
        if ('speechSynthesis' in window) {
            this.loadVoices();
            window.speechSynthesis.onvoiceschanged = () => this.loadVoices();
            this.setupControls();
        } else {
            console.warn('Speech synthesis not supported in this browser');
        }
    }

    loadVoices() {
        this.voices = window.speechSynthesis.getVoices();
        this.populateVoiceSelect();
        
        // Set default voices
        if (this.voices.length > 0) {
            this.modelAVoice = this.voices.find(v => v.name.includes('Google UK English Female')) || this.voices[0];
            this.modelBVoice = this.voices.find(v => v.name.includes('Google UK English Male')) || this.voices[1] || this.voices[0];
        }
    }

    populateVoiceSelect() {
        const voiceSelect = document.getElementById('voiceSelect');
        if (voiceSelect) {
            voiceSelect.innerHTML = '';
            this.voices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                voiceSelect.appendChild(option);
            });
        }
    }

    setupControls() {
        const ttsToggle = document.getElementById('ttsToggle');
        const voiceSelect = document.getElementById('voiceSelect');
        const speechRate = document.getElementById('speechRate');

        if (ttsToggle) {
            ttsToggle.addEventListener('click', () => this.toggleTTS());
        }

        if (voiceSelect) {
            voiceSelect.addEventListener('change', (e) => {
                this.currentVoice = this.voices[e.target.value];
            });
        }

        if (speechRate) {
            speechRate.addEventListener('input', (e) => {
                this.speechRate = parseFloat(e.target.value);
            });
        }
    }

    toggleTTS() {
        this.enabled = !this.enabled;
        const toggleBtn = document.getElementById('ttsToggle');
        const controlsDiv = document.getElementById('ttsControls');
        
        if (this.enabled) {
            toggleBtn.textContent = '🔊 TTS On';
            toggleBtn.className = 'px-3 py-1 bg-green-500 text-white rounded text-sm';
            controlsDiv.classList.remove('hidden');
        } else {
            toggleBtn.textContent = '🔇 TTS Off';
            toggleBtn.className = 'px-3 py-1 bg-gray-500 text-white rounded text-sm';
            controlsDiv.classList.add('hidden');
            this.stopSpeech();
        }
    }

    speakTurn(turn) {
        if (!this.enabled || !this.voices.length) return;

        // Stop any current speech
        this.stopSpeech();

        // Select voice based on turn number (Model A vs Model B)
        const voice = turn.turn_number % 2 === 1 ? this.modelAVoice : this.modelBVoice;

        this.speakText(turn.message, voice);
    }

    speakText(text, voice = null) {
        if (!text || !this.enabled) return;

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = voice || this.currentVoice || this.voices[0];
        utterance.rate = this.speechRate;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;

        utterance.onstart = () => {
            this.currentSpeech = utterance;
        };

        utterance.onend = () => {
            this.currentSpeech = null;
        };

        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event.error);
            this.currentSpeech = null;
        };

        window.speechSynthesis.speak(utterance);
    }

    stopSpeech() {
        if (this.currentSpeech) {
            window.speechSynthesis.cancel();
            this.currentSpeech = null;
        }
    }

    pauseSpeech() {
        if (this.currentSpeech) {
            window.speechSynthesis.pause();
        }
    }

    resumeSpeech() {
        if (this.currentSpeech) {
            window.speechSynthesis.resume();
        }
    }

    isEnabled() {
        return this.enabled;
    }
}

// Initialize TTS manager
window.ttsManager = new TTSManager();
```

## Development Setup

### Environment Setup
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
```bash
# Copy and edit .env file
cp .env.example .env
# Add your API keys to .env
```

3. **Initialize database**:
```bash
# Create database tables
python -c "from dashboard.app import app, db; app.app_context().push(); db.create_all()"
```

4. **Run development server**:
```bash
# Start Flask dashboard
cd dashboard
python app.py

# Dashboard will be available at http://localhost:5000
```

### Integration with CLI Tool
The CLI tool needs to be modified to save conversations to the database and emit WebSocket events:

```python
# In lll-talk.py, add database integration
from dashboard.models import Conversation, Turn, db
from dashboard.websocket_handlers import emit_conversation_update

def save_conversation_to_db(self, conversation_id, metrics):
    """Save conversation to database"""
    # Create conversation record
    conversation = Conversation(
        model_1=self.model_1,
        model_2=self.model_2,
        mode=self.mode,
        window_size=self.window_size,
        template=self.template,
        initial_prompt=self.initial_prompt,
        ai_aware_mode=self.ai_aware_mode,
        total_turns=len(self.conversation_history),
        total_cost=metrics.total_cost,
        total_input_tokens=metrics.total_input_tokens,
        total_output_tokens=metrics.total_output_tokens,
        status='completed'
    )
    
    db.session.add(conversation)
    db.session.flush()  # Get the ID
    
    # Save turns
    for turn in self.conversation_history:
        turn_record = Turn(
            conversation_id=conversation.id,
            turn_number=turn.turn_number,
            speaker=turn.speaker,
            message=turn.message,
            timestamp=datetime.fromtimestamp(turn.timestamp),
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            response_time=turn.response_time
        )
        db.session.add(turn_record)
    
    db.session.commit()
    
    # Trigger AI analysis
    analyzer = ConversationAnalyzer(api_key=self.api_key)
    analyzer.analyze_conversation(conversation.id)
    
    return conversation.id
```

### Background Job Processing
For processing AI analysis of conversations:

```python
# scripts/process_conversations.py
import time
import logging
from dashboard.conversation_analyzer import ConversationAnalyzer
from dashboard.models import Conversation, db

def process_pending_conversations():
    """Background job to process conversations without AI analysis"""
    analyzer = ConversationAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Find conversations without analysis
    pending = Conversation.query.filter(
        Conversation.summary.is_(None)
    ).limit(10).all()
    
    for conversation in pending:
        try:
            logging.info(f"Processing conversation {conversation.id}")
            success = analyzer.analyze_conversation(conversation.id)
            
            if success:
                logging.info(f"Successfully analyzed conversation {conversation.id}")
            else:
                logging.error(f"Failed to analyze conversation {conversation.id}")
                
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error processing conversation {conversation.id}: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_pending_conversations()
```

## Cost Estimation

### AI Generation Costs
- **Summary generation**: ~150 tokens per conversation
- **Keyword extraction**: ~100 tokens per conversation
- **Total per conversation**: ~250 tokens ≈ $0.0005 (using GPT-3.5-turbo)
- **Monthly cost** (100 conversations): ~$0.05

### Infrastructure Costs
- **Development**: Local SQLite database (free)
- **Production**: Consider PostgreSQL + Redis for scaling
- **Hosting**: Flask app can run on minimal servers ($5-10/month)

## Security Considerations

1. **API Key Management**: Store API keys securely in environment variables
2. **Input Validation**: Sanitize all user inputs and search queries
3. **Rate Limiting**: Implement rate limiting on API endpoints
4. **Authentication**: Add user authentication if needed
5. **Data Privacy**: Ensure conversation data is properly secured

## Future Enhancements

### Near-term Enhancements
1. **Real-time Collaboration**: Multiple users viewing same conversation
2. **Advanced Analytics**: Conversation quality trends, model performance
3. **Export Features**: PDF, email sharing, public links
4. **API Integration**: RESTful API for external access

### 3-Model Symposium (Experimental)
**Concept**: Add a third model as moderator/facilitator to enhance 2-model conversations

**Quick Win Approach - Smart Moderator**:
```python
# Moderator intervention every N turns
if turn_number % 5 == 0:
    moderator_prompt = f"""
    You're moderating this conversation between two AI models.
    Recent discussion: {last_5_turns}
    
    Please:
    1. Summarize key points made
    2. Identify areas of agreement/disagreement  
    3. Ask a clarifying question to advance discussion
    """
```

**Implementation Phases**:
1. **Phase 1**: Periodic moderator summaries (every 5 turns)
2. **Phase 2**: Context-aware interventions (detect repetition, stagnation)
3. **Phase 3**: Full 3-way symposium with dynamic turn management

**Database Schema Extensions**:
```sql
-- Support for 3+ models
ALTER TABLE conversations ADD COLUMN model_3 TEXT;
ALTER TABLE conversations ADD COLUMN moderator_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE turns ADD COLUMN model_role TEXT; -- 'participant', 'moderator'
ALTER TABLE turns ADD COLUMN intervention_type TEXT; -- 'summary', 'question', 'clarification'
```

**UI Enhancements**:
- Moderator messages visually distinct (different color, icon)
- "Moderator insights" panel showing key summaries
- Toggle to enable/disable moderator mode
- Moderator intervention frequency settings

**Use Cases**:
- **Expert Panel**: 3 different domain experts discussing a topic
- **Peer Review**: Author, reviewer 1, reviewer 2 format
- **Teaching Triangle**: Teacher, student, observer roles
- **Philosophical Triad**: Thesis, antithesis, synthesis approach

**Templates for 3-Model Discussions**:
```python
"expert_panel": {
    "model_1": {"role": "technical_expert", "system_prompt": "..."},
    "model_2": {"role": "business_expert", "system_prompt": "..."},
    "model_3": {"role": "user_advocate", "system_prompt": "..."}
},
"peer_review": {
    "model_1": {"role": "author", "system_prompt": "..."},
    "model_2": {"role": "reviewer_1", "system_prompt": "..."},
    "model_3": {"role": "reviewer_2", "system_prompt": "..."}
}
```

### Long-term Enhancements
5. **Mobile App**: Native mobile apps for iOS/Android
6. **Cloud Deployment**: Docker containers, CI/CD pipelines
7. **Multi-Model Orchestration**: Support for 4+ models in complex scenarios

This specification provides a comprehensive foundation for building a modern, feature-rich Flask dashboard that transforms the CLI conversation tool into a powerful web application for analyzing and exploring AI-to-AI conversations.