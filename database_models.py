"""
SQLAlchemy Database Models for LLM Conversations

Contains the database schema for storing conversations, turns, and metrics.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json

db = SQLAlchemy()


class Conversation(db.Model):
    """Main conversation metadata."""
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Configuration
    models = db.Column(db.Text, nullable=False)  # JSON array of model names
    model_count = db.Column(db.Integer, nullable=False, default=2)
    mode = db.Column(db.String(20), nullable=False)  # 'full', 'sliding', 'cache', 'sliding_cache'
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
    status = db.Column(db.String(20), default='active')  # 'active', 'completed', 'error'
    
    # AI-generated content
    summary = db.Column(db.Text)
    keywords = db.Column(db.Text)  # JSON array of keywords
    topic_category = db.Column(db.String(50))
    summary_generated_at = db.Column(db.DateTime)
    keywords_generated_at = db.Column(db.DateTime)
    summary_model = db.Column(db.String(50))
    keywords_model = db.Column(db.String(50))
    
    # Relationships
    turns = db.relationship('Turn', backref='conversation', lazy=True, cascade='all, delete-orphan')
    quality_metrics = db.relationship('QualityMetric', backref='conversation', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert conversation to dictionary."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'models': self.get_models(),
            'model_count': self.model_count,
            'mode': self.mode,
            'window_size': self.window_size,
            'template': self.template,
            'initial_prompt': self.initial_prompt,
            'ai_aware_mode': self.ai_aware_mode,
            'total_turns': self.total_turns,
            'total_cost': float(self.total_cost) if self.total_cost else 0.0,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'avg_response_time': float(self.avg_response_time) if self.avg_response_time else 0.0,
            'status': self.status,
            'summary': self.summary,
            'keywords': json.loads(self.keywords) if self.keywords else [],
            'topic_category': self.topic_category,
            'summary_generated_at': self.summary_generated_at.isoformat() if self.summary_generated_at else None,
            'keywords_generated_at': self.keywords_generated_at.isoformat() if self.keywords_generated_at else None,
            'summary_model': self.summary_model,
            'keywords_model': self.keywords_model
        }
    
    def set_models(self, models_list):
        """Set models from a list."""
        self.models = json.dumps(models_list) if models_list else None
        self.model_count = len(models_list) if models_list else 0
    
    def get_models(self):
        """Get models as a list."""
        return json.loads(self.models) if self.models else []
    
    def set_keywords(self, keywords_list):
        """Set keywords from a list."""
        self.keywords = json.dumps(keywords_list) if keywords_list else None
    
    def get_keywords(self):
        """Get keywords as a list."""
        return json.loads(self.keywords) if self.keywords else []
    
    def __repr__(self):
        models = self.get_models()
        if len(models) <= 3:
            model_display = " vs ".join(models)
        else:
            model_display = f"{models[0]} vs {models[1]} (+{len(models)-2} more)"
        return f'<Conversation {self.id}: {model_display}>'


class Turn(db.Model):
    """Individual conversation turns."""
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
        """Convert turn to dictionary."""
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
    
    def __repr__(self):
        return f'<Turn {self.id}: {self.conversation_id}-{self.turn_number}>'


class QualityMetric(db.Model):
    """Optional quality metrics for conversations."""
    __tablename__ = 'quality_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    turn_number = db.Column(db.Integer)
    coherence_score = db.Column(db.Numeric(5, 3))
    topic_drift_score = db.Column(db.Numeric(5, 3))
    repetition_rate = db.Column(db.Numeric(5, 3))
    
    def to_dict(self):
        """Convert quality metric to dictionary."""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'turn_number': self.turn_number,
            'coherence_score': float(self.coherence_score) if self.coherence_score else None,
            'topic_drift_score': float(self.topic_drift_score) if self.topic_drift_score else None,
            'repetition_rate': float(self.repetition_rate) if self.repetition_rate else None
        }
    
    def __repr__(self):
        return f'<QualityMetric {self.id}: {self.conversation_id}-{self.turn_number}>'


def create_tables(app):
    """Create all database tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")


def init_database(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    return db


# Database utility functions
def get_conversation_by_id(conversation_id):
    """Get a conversation by ID."""
    return Conversation.query.get(conversation_id)


def get_all_conversations(page=1, per_page=20):
    """Get all conversations with pagination."""
    return Conversation.query.order_by(Conversation.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )


def get_conversations_by_template(template, page=1, per_page=20):
    """Get conversations by template."""
    return Conversation.query.filter_by(template=template).order_by(
        Conversation.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)


def get_conversations_by_status(status, page=1, per_page=20):
    """Get conversations by status."""
    return Conversation.query.filter_by(status=status).order_by(
        Conversation.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)


def search_conversations(query, page=1, per_page=20):
    """Search conversations by summary or keywords."""
    search_pattern = f'%{query}%'
    return Conversation.query.filter(
        db.or_(
            Conversation.summary.ilike(search_pattern),
            Conversation.keywords.ilike(search_pattern),
            Conversation.initial_prompt.ilike(search_pattern)
        )
    ).order_by(Conversation.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )


def get_conversation_turns(conversation_id):
    """Get all turns for a conversation."""
    return Turn.query.filter_by(conversation_id=conversation_id).order_by(Turn.turn_number).all()


def get_conversation_quality_metrics(conversation_id):
    """Get quality metrics for a conversation."""
    return QualityMetric.query.filter_by(conversation_id=conversation_id).order_by(QualityMetric.turn_number).all()


def get_database_stats():
    """Get database statistics."""
    return {
        'total_conversations': Conversation.query.count(),
        'completed_conversations': Conversation.query.filter_by(status='completed').count(),
        'active_conversations': Conversation.query.filter_by(status='active').count(),
        'total_turns': Turn.query.count(),
        'total_quality_metrics': QualityMetric.query.count(),
        'avg_turns_per_conversation': db.session.query(db.func.avg(Conversation.total_turns)).scalar() or 0,
        'avg_cost_per_conversation': db.session.query(db.func.avg(Conversation.total_cost)).scalar() or 0,
        'templates_used': db.session.query(Conversation.template).distinct().count(),
        'models_used': db.session.query(
            db.func.count(db.distinct(Conversation.model_1)) + 
            db.func.count(db.distinct(Conversation.model_2))
        ).scalar() or 0
    }