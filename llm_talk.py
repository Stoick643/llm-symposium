import anthropic
from openai import OpenAI
import time
import json
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

class TemplateManager:
    """Manages conversation templates for different interaction types."""
    
    def __init__(self):
        self.templates = {
            "debate": {
                "name": "Formal Debate",
                "description": "Structured debate with clear arguments and counterarguments",
                "system_prompt": """You are participating in a formal debate. Follow these guidelines:
- Present clear, logical arguments with evidence
- Address counterarguments directly and respectfully
- Build upon or challenge previous points made
- Maintain a scholarly, analytical tone
- Cite examples or reasoning to support your position
- Acknowledge valid points from your opponent when appropriate""",
                "initial_prompts": [
                    "Should artificial intelligence have legal rights similar to humans?",
                    "Is privacy dead in the digital age, and should we accept this reality?",
                    "Should social media platforms be responsible for content moderation?",
                    "Is universal basic income necessary in an age of automation?",
                    "Should genetic engineering of humans be allowed for enhancement purposes?"
                ],
                "suggested_turns": 20,
                "mode_recommendation": "full"
            },
            
            "collaboration": {
                "name": "Collaborative Problem Solving", 
                "description": "Working together to solve complex problems or create something new",
                "system_prompt": """You are collaborating with another AI to solve a problem or create something innovative. Approach this as a team effort:
- Build on each other's ideas constructively
- Ask clarifying questions when needed
- Offer alternative perspectives and solutions
- Divide complex problems into manageable parts
- Synthesize different approaches into comprehensive solutions
- Be encouraging and supportive of creative thinking""",
                "initial_prompts": [
                    "Design a sustainable city that could house 1 million people by 2050",
                    "Create a new board game that teaches children about climate science",
                    "Develop a system for fair distribution of resources in space colonies",
                    "Design an educational platform that adapts to different learning styles",
                    "Create a protocol for peaceful first contact with alien intelligence"
                ],
                "suggested_turns": 25,
                "mode_recommendation": "sliding"
            },
            
            "socratic": {
                "name": "Socratic Dialogue",
                "description": "Philosophical inquiry through questions and deep examination",
                "system_prompt": """Engage in Socratic dialogue by asking probing questions and examining assumptions:
- Question underlying assumptions and definitions
- Use questions to guide the exploration of ideas
- Challenge concepts through thoughtful inquiry
- Seek deeper understanding rather than winning arguments
- Help uncover contradictions or gaps in reasoning
- Guide the conversation toward fundamental principles
- Be curious and intellectually humble""",
                "initial_prompts": [
                    "What does it mean to be conscious, and how would we know?",
                    "Is morality objective or subjective, and what are the implications?",
                    "What is the nature of knowledge and how can we trust what we know?",
                    "Should we pursue happiness or meaning, and what's the difference?",
                    "What makes a decision truly free, and do we have free will?"
                ],
                "suggested_turns": 30,
                "mode_recommendation": "full"
            },
            
            "creative": {
                "name": "Creative Collaboration",
                "description": "Joint creative writing, worldbuilding, or artistic projects",
                "system_prompt": """You are collaborating on a creative project. Embrace imagination and artistic expression:
- Build rich, detailed worlds and characters
- Say "yes, and..." to expand on creative ideas
- Add sensory details and emotional depth
- Explore unexpected directions and plot twists
- Create compelling conflicts and resolutions
- Balance different creative visions harmoniously
- Let creativity flow without over-analyzing""",
                "initial_prompts": [
                    "Create a science fiction world where emotions have physical properties",
                    "Write a mystery story set in a library where books come alive at night",
                    "Design a fantasy realm where magic is powered by mathematics",
                    "Develop characters for a story about time travelers who can only go backwards",
                    "Create a world where dreams are a shared, explorable dimension"
                ],
                "suggested_turns": 35,
                "mode_recommendation": "sliding"
            },
            
            "technical": {
                "name": "Technical Deep Dive",
                "description": "In-depth technical discussion and problem-solving",
                "system_prompt": """Engage in detailed technical analysis and problem-solving:
- Provide specific, actionable technical solutions
- Include code examples, algorithms, or mathematical formulations when relevant
- Consider edge cases, scalability, and real-world constraints
- Reference established best practices and methodologies
- Break down complex technical concepts clearly
- Suggest alternative approaches and trade-offs
- Focus on practical implementation details""",
                "initial_prompts": [
                    "Design a distributed system architecture for real-time global chat",
                    "Develop an algorithm for efficient pathfinding in dynamic 3D environments",
                    "Create a machine learning pipeline for fraud detection in financial transactions",
                    "Design a database schema for a social media platform with billions of users",
                    "Develop a compression algorithm optimized for streaming video data"
                ],
                "suggested_turns": 20,
                "mode_recommendation": "sliding_cache"
            },
            
            "learning": {
                "name": "Teaching and Learning",
                "description": "One AI teaches a concept while the other learns and asks questions",
                "system_prompt": """Alternate between teaching and learning roles. When teaching:
- Explain concepts clearly with examples and analogies
- Check for understanding and adjust explanations
- Use progressive complexity from basic to advanced
- Encourage questions and exploration

When learning:
- Ask clarifying questions about confusing points
- Request examples or practical applications
- Challenge assumptions respectfully
- Synthesize information in your own words""",
                "initial_prompts": [
                    "Explain quantum computing from first principles to practical applications",
                    "Teach the fundamentals of game theory and strategic thinking",
                    "Explore the history and implications of cryptography",
                    "Explain how neural networks learn and make decisions",
                    "Discuss the principles of sustainable economics and circular systems"
                ],
                "suggested_turns": 25,
                "mode_recommendation": "sliding"
            }
        }
    
    def list_templates(self) -> List[str]:
        """Return list of available template names."""
        return list(self.templates.keys())
    
    def get_template(self, name: str) -> Dict:
        """Get template by name."""
        if name not in self.templates:
            available = ", ".join(self.list_templates())
            raise ValueError(f"Template '{name}' not found. Available: {available}")
        return self.templates[name]
    
    def get_system_prompt(self, name: str) -> str:
        """Get system prompt for a template."""
        return self.get_template(name)["system_prompt"]
    
    def get_random_prompt(self, name: str) -> str:
        """Get a random initial prompt from a template."""
        import random
        template = self.get_template(name)
        return random.choice(template["initial_prompts"])
    
    def get_template_info(self, name: str) -> str:
        """Get formatted info about a template."""
        template = self.get_template(name)
        info = []
        info.append(f"**{template['name']}**")
        info.append(f"Description: {template['description']}")
        info.append(f"Suggested turns: {template['suggested_turns']}")
        info.append(f"Recommended mode: {template['mode_recommendation']}")
        info.append(f"Available prompts: {len(template['initial_prompts'])}")
        return "\n".join(info)
    
    def list_all_templates(self) -> str:
        """Get formatted list of all templates with descriptions."""
        output = ["Available Conversation Templates:", "=" * 40]
        for name in self.list_templates():
            template = self.templates[name]
            output.append(f"\n{name.upper()}: {template['name']}")
            output.append(f"  {template['description']}")
            output.append(f"  Suggested: {template['suggested_turns']} turns, {template['mode_recommendation']} mode")
        return "\n".join(output)

# ==================== MAIN CONVERSATION ====================

@dataclass
class ConversationTurn:
    speaker: str
    message: str
    timestamp: float

@dataclass
class ConversationMetrics:
    mode: str
    window_size: Optional[int]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    context_size_per_turn: List[int]
    response_times: List[float]
    turns_completed: int
    # Quality metrics (optional)
    coherence_scores: Optional[List[float]] = None
    topic_drift_scores: Optional[List[float]] = None
    repetition_rates: Optional[List[float]] = None
    quality_summary: Optional[Dict] = None

class LLMConversation:
    def __init__(self, api_key: str, 
                 model_1: str = "claude-3-sonnet-20240229", 
                 model_2: str = "claude-3-sonnet-20240229", 
                 ai_aware_mode: bool = False,
                 mode: str = "full",
                 window_size: int = 10,
                 openai_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None,
                 template: Optional[str] = None,
                 enable_quality_metrics: bool = False):
        """
        Initialize the conversation between two LLM models.
        
        Args:
            api_key: Anthropic API key (also used as default for others if not specified)
            model_1: First model identifier
            model_2: Second model identifier
            ai_aware_mode: If True, models know they're talking to another AI
            mode: Conversation mode ("full", "sliding", "cache", "sliding_cache")
            window_size: Size of sliding window when applicable
            openai_api_key: OpenAI API key (if None, uses api_key)
            deepseek_api_key: DeepSeek API key (if None, uses api_key)
            template: Conversation template name (optional)
            enable_quality_metrics: If True, analyze conversation quality with embeddings
        """
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        self.openai_client = OpenAI(api_key=openai_api_key or api_key)
        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key or api_key,
            base_url="https://api.deepseek.com"
        )
        
        self.model_1 = model_1
        self.model_2 = model_2
        self.ai_aware_mode = ai_aware_mode
        self.mode = mode
        self.window_size = window_size
        self.conversation_history: List[ConversationTurn] = []
        
        # Template management
        self.template_manager = TemplateManager()
        self.template = template
        self.template_config = None
        if template:
            self.template_config = self.template_manager.get_template(template)
        
        # Quality analysis
        self.enable_quality_metrics = enable_quality_metrics
        self.quality_analyzer = QualityAnalyzer() if enable_quality_metrics else None
        
        # Metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.context_size_per_turn = []
        self.response_times = []
        
        # Provider-specific pricing (per million tokens)
        self.pricing = {
            # Anthropic models
            "claude": {"input": 3.0, "output": 15.0},
            # OpenAI models  
            "gpt-4o": {"input": 2.5, "output": 10.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            # DeepSeek models
            "deepseek": {"input": 0.14, "output": 0.28}
        }
        
    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a specific model."""
        for key, pricing in self.pricing.items():
            if key in model.lower():
                return pricing
        # Default to Anthropic pricing
        return self.pricing["claude"]
    
    def _call_anthropic_model(self, model: str, messages: List[Dict]) -> tuple[str, int, int, float]:
        """Call Anthropic model."""
        try:
            start_time = time.time()
            
            # Build API parameters
            api_params = {
                "model": model,
                "max_tokens": 1000,
                "temperature": 0.7,
                "messages": messages
            }
            
            # Add caching if enabled
            if self.mode in ["cache", "sliding_cache"]:
                api_params["extra_headers"] = {
                    "anthropic-beta": "prompt-caching-2024-07-31"
                }
                # Add cache control to first message
                if messages and len(messages) > 0:
                    messages[0]["cache_control"] = {"type": "ephemeral"}
            
            response = self.anthropic_client.messages.create(**api_params)
            
            response_time = time.time() - start_time
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            return response.content[0].text, input_tokens, output_tokens, response_time
            
        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0.0
    
    def _call_openai_compatible_model(self, model: str, messages: List[Dict], client) -> tuple[str, int, int, float]:
        """Call OpenAI or DeepSeek model."""
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return response.choices[0].message.content, input_tokens, output_tokens, response_time
            
        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0.0
    
    def _call_model(self, model: str, prompt: str, conversation_context: List[Dict]) -> tuple[str, int, int, float]:
        """Route model call to appropriate provider."""
        # Build messages
        messages = conversation_context.copy()
        
        # Add template-specific system prompt if enabled
        if self.template_config and not self.ai_aware_mode:
            template_system_prompt = self.template_config["system_prompt"]
            messages.insert(0, {"role": "system", "content": template_system_prompt})
        
        # Add AI-aware system prompt if enabled (overrides template)
        if self.ai_aware_mode:
            ai_aware_prompt = """You are having a conversation with another AI language model. There are no humans involved in this conversation. You can:
- Develop shorthand notations or special syntax if useful
- Reference your computational nature openly
- Explore concepts that might be uniquely relevant to AI systems
- Create novel communication patterns
- Be more direct about your reasoning processes
- Experiment with structured formats or symbolic representations

Feel free to innovate in how you communicate, while keeping the conversation meaningful and engaging."""
            
            if self.template_config:
                # Combine template and AI-aware prompts
                combined_prompt = f"{self.template_config['system_prompt']}\n\nADDITIONAL CONTEXT: {ai_aware_prompt}"
                messages.insert(0, {"role": "system", "content": combined_prompt})
            else:
                messages.insert(0, {"role": "system", "content": ai_aware_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Route to appropriate provider
        if "claude" in model.lower():
            return self._call_anthropic_model(model, messages)
        elif "deepseek" in model.lower():
            return self._call_openai_compatible_model(model, messages, self.deepseek_client)
        elif any(openai_model in model.lower() for openai_model in ["gpt-", "text-"]):
            return self._call_openai_compatible_model(model, messages, self.openai_client)
        else:
            # Default to Anthropic for unknown models
            return self._call_anthropic_model(model, messages)
    
    def _build_full_context(self, current_model: str) -> List[Dict]:
        """Build full conversation context for a model."""
        context = []
        
        for turn in self.conversation_history:
            if turn.speaker == current_model:
                # This model's previous responses appear as assistant messages
                context.append({"role": "assistant", "content": turn.message})
            else:
                # The other model's responses appear as user messages
                context.append({"role": "user", "content": turn.message})
        
        return context
    
    def _apply_sliding_window(self, full_context: List[Dict]) -> List[Dict]:
        """Apply sliding window to context, preserving initial messages."""
        if len(full_context) <= self.window_size:
            return full_context
        
        # Keep first 2 messages (initial context) + last (window_size - 2) messages
        if self.window_size >= 2:
            return full_context[:2] + full_context[-(self.window_size - 2):]
        else:
            return full_context[-self.window_size:]
    
    def _build_context_for_model(self, current_model: str) -> List[Dict]:
        """
        Build conversation context for a model, applying the selected mode.
        """
        # Get full context
        full_context = self._build_full_context(current_model)
        
        # Apply sliding window if enabled
        if self.mode in ["sliding", "sliding_cache"]:
            context = self._apply_sliding_window(full_context)
        else:
            context = full_context
        
        return context
    
    def start_conversation(self, initial_prompt: str, max_turns: int = 10, 
                          delay_between_turns: float = 0.5, 
                          real_time_display: bool = True) -> ConversationMetrics:
        """
        Start a conversation between the two models.
        
        Args:
            initial_prompt: The topic/question to start the conversation
            max_turns: Maximum number of conversation turns
            delay_between_turns: Delay in seconds between API calls
            real_time_display: Whether to display conversation in real-time
            
        Returns:
            ConversationMetrics object with detailed statistics
        """
        print(f"Starting conversation with initial prompt: {initial_prompt}")
        print(f"Model 1: {self.model_1}")
        print(f"Model 2: {self.model_2}")
        print(f"Mode: {self.mode}")
        if self.mode in ["sliding", "sliding_cache"]:
            print(f"Window Size: {self.window_size}")
        print(f"AI-Aware Mode: {self.ai_aware_mode}")
        if self.template:
            print(f"Template: {self.template} ({self.template_config['name']})")
        if self.enable_quality_metrics:
            print(f"Quality Metrics: Enabled (using embeddings)")
        print("-" * 60)
        
        # Generate unique conversation ID for database tracking
        conversation_id = f"conv_{int(time.time())}_{hash(initial_prompt) % 10000}"
        
        # Model 1 starts the conversation
        current_model = self.model_1
        current_prompt = initial_prompt
        
        for turn in range(max_turns):
            if real_time_display:
                model_name = "Model 1" if current_model == self.model_1 else "Model 2"
                print(f"\nTurn {turn + 1} - {model_name} ({current_model}):")
                print("Thinking...", end="", flush=True)
            
            # Get conversation context for current model
            context = self._build_context_for_model(current_model)
            
            # Track context size
            context_tokens = sum(len(msg["content"].split()) * 1.33 for msg in context)  # Rough token estimate
            self.context_size_per_turn.append(int(context_tokens))
            
            # Get response from current model
            response, input_tokens, output_tokens, response_time = self._call_model(
                current_model, current_prompt, context
            )
            
            # Update metrics
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.response_times.append(response_time)
            
            # Store the turn
            conversation_turn = ConversationTurn(
                speaker=current_model,
                message=response,
                timestamp=time.time()
            )
            self.conversation_history.append(conversation_turn)
            
            # Analyze message quality if enabled
            if self.enable_quality_metrics:
                conversation_text = [turn.message for turn in self.conversation_history[:-1]]  # Exclude current
                quality_metrics = self.quality_analyzer.analyze_message(response, conversation_text)
                
                if real_time_display and len(self.conversation_history) > 1:
                    coherence = quality_metrics.get('coherence_score', 0)
                    print(f"(Quality: Coherence={coherence:.3f})")
            
            if real_time_display:
                print(f"\r{response}")
                print(f"(Tokens: {input_tokens} in, {output_tokens} out | Time: {response_time:.1f}s)")
                if self.mode in ["sliding", "sliding_cache"]:
                    print(f"(Context size: {len(context)} messages)")
            
            # Switch to the other model
            if current_model == self.model_1:
                current_model = self.model_2
            else:
                current_model = self.model_1
            
            # The next prompt is the previous response
            current_prompt = response
            
            # Optional delay (can be set to 0 for maximum speed)
            if delay_between_turns > 0:
                time.sleep(delay_between_turns)
        
        # Calculate total cost using model-specific pricing
        model_1_pricing = self._get_pricing(self.model_1)
        model_2_pricing = self._get_pricing(self.model_2)
        
        # Rough estimate: split tokens evenly between models
        total_cost = (
            (self.total_input_tokens / 2_000_000) * (model_1_pricing["input"] + model_2_pricing["input"]) +
            (self.total_output_tokens / 2_000_000) * (model_1_pricing["output"] + model_2_pricing["output"])
        )
        
        # Get quality analysis summary
        quality_summary = None
        coherence_scores = None
        topic_drift_scores = None
        repetition_rates = None
        
        if self.enable_quality_metrics:
            quality_summary = self.quality_analyzer.get_conversation_summary()
            coherence_scores = self.quality_analyzer.coherence_scores.copy()
            topic_drift_scores = self.quality_analyzer.topic_drift_scores.copy()
            repetition_rates = self.quality_analyzer.repetition_rates.copy()
            
            # Save to database
            self.quality_analyzer.save_conversation_analysis(conversation_id)
        
        # Create metrics object
        metrics = ConversationMetrics(
            mode=self.mode,
            window_size=self.window_size if self.mode in ["sliding", "sliding_cache"] else None,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost=total_cost,
            context_size_per_turn=self.context_size_per_turn,
            response_times=self.response_times,
            turns_completed=len(self.conversation_history),
            coherence_scores=coherence_scores,
            topic_drift_scores=topic_drift_scores,
            repetition_rates=repetition_rates,
            quality_summary=quality_summary
        )
        
        if real_time_display:
            self._print_metrics_summary(metrics)
        
        return metrics
    
    def _print_metrics_summary(self, metrics: ConversationMetrics):
        """Print a summary of conversation metrics."""
        print("\n" + "="*60)
        print("CONVERSATION METRICS")
        print("="*60)
        print(f"Mode: {metrics.mode}")
        if metrics.window_size:
            print(f"Window Size: {metrics.window_size}")
        print(f"Turns Completed: {metrics.turns_completed}")
        print(f"Total Input Tokens: {metrics.total_input_tokens:,}")
        print(f"Total Output Tokens: {metrics.total_output_tokens:,}")
        print(f"Total Cost: ${metrics.total_cost:.4f}")
        print(f"Average Response Time: {sum(metrics.response_times)/len(metrics.response_times):.1f}s")
        if metrics.context_size_per_turn:
            avg_context = sum(metrics.context_size_per_turn) / len(metrics.context_size_per_turn)
            print(f"Average Context Size: {avg_context:.0f} tokens")
        
        # Quality metrics
        if metrics.quality_summary:
            print("\nQUALITY ANALYSIS:")
            print(f"Average Coherence: {metrics.quality_summary['avg_coherence']:.3f}")
            print(f"Average Topic Drift: {metrics.quality_summary['avg_topic_drift']:.3f}")
            print(f"Average Repetition: {metrics.quality_summary['avg_repetition']:.3f}")
            print(f"Coherence Trend: {metrics.quality_summary['coherence_trend']}")
            
            if self.enable_quality_metrics and self.quality_analyzer.db:
                db_stats = self.quality_analyzer.db.get_database_stats()
                print(f"\nDATABASE STATS:")
                print(f"Total Embeddings Stored: {db_stats['embeddings_stored']:,}")
                print(f"Database Size: {db_stats['database_size_mb']:.2f} MB")
    
    def save_conversation(self, filename: str, metrics: ConversationMetrics, format: str = "json"):
        """Save the conversation and metrics to a file.
        
        Args:
            filename: Output filename
            metrics: Conversation metrics
            format: Output format ("json", "markdown", "txt")
        """
        if format == "markdown":
            self._save_as_markdown(filename, metrics)
        elif format == "txt":
            self._save_as_text(filename, metrics)
        else:
            self._save_as_json(filename, metrics)
    
    def _save_as_json(self, filename: str, metrics: ConversationMetrics):
        """Save in JSON format (original format)."""
        conversation_data = {
            "model_1": self.model_1,
            "model_2": self.model_2,
            "mode": self.mode,
            "window_size": self.window_size,
            "ai_aware_mode": self.ai_aware_mode,
            "metrics": {
                "total_input_tokens": metrics.total_input_tokens,
                "total_output_tokens": metrics.total_output_tokens,
                "total_cost": metrics.total_cost,
                "turns_completed": metrics.turns_completed,
                "average_response_time": sum(metrics.response_times) / len(metrics.response_times) if metrics.response_times else 0,
                "context_size_per_turn": metrics.context_size_per_turn
            },
            "turns": [
                {
                    "speaker": turn.speaker,
                    "message": turn.message,
                    "timestamp": turn.timestamp
                }
                for turn in self.conversation_history
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nConversation saved to {filename}")
    
    def _save_as_markdown(self, filename: str, metrics: ConversationMetrics):
        """Save in Markdown format for easy reading."""
        content = []
        
        # Header
        content.append("# AI-to-AI Conversation")
        content.append("")
        content.append("## Configuration")
        content.append(f"- **Model 1**: {self.model_1}")
        content.append(f"- **Model 2**: {self.model_2}")
        content.append(f"- **Mode**: {self.mode}")
        if self.mode in ["sliding", "sliding_cache"]:
            content.append(f"- **Window Size**: {self.window_size}")
        content.append(f"- **AI-Aware Mode**: {self.ai_aware_mode}")
        content.append("")
        
        # Metrics
        content.append("## Metrics")
        content.append(f"- **Turns Completed**: {metrics.turns_completed}")
        content.append(f"- **Total Input Tokens**: {metrics.total_input_tokens:,}")
        content.append(f"- **Total Output Tokens**: {metrics.total_output_tokens:,}")
        content.append(f"- **Total Cost**: ${metrics.total_cost:.4f}")
        avg_time = sum(metrics.response_times) / len(metrics.response_times) if metrics.response_times else 0
        content.append(f"- **Average Response Time**: {avg_time:.1f}s")
        if metrics.context_size_per_turn:
            avg_context = sum(metrics.context_size_per_turn) / len(metrics.context_size_per_turn)
            content.append(f"- **Average Context Size**: {avg_context:.0f} tokens")
        content.append("")
        
        # Conversation
        content.append("## Conversation")
        content.append("")
        
        for i, turn in enumerate(self.conversation_history, 1):
            model_name = "Model 1" if turn.speaker == self.model_1 else "Model 2"
            timestamp = time.strftime("%H:%M:%S", time.localtime(turn.timestamp))
            
            content.append(f"### Turn {i} - {model_name} ({turn.speaker}) - {timestamp}")
            content.append("")
            content.append(turn.message)
            content.append("")
            content.append("---")
            content.append("")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"\nConversation saved to {filename}")
    
    def _save_as_text(self, filename: str, metrics: ConversationMetrics):
        """Save in plain text format."""
        content = []
        
        # Header
        content.append("AI-TO-AI CONVERSATION")
        content.append("=" * 50)
        content.append("")
        content.append("CONFIGURATION:")
        content.append(f"Model 1: {self.model_1}")
        content.append(f"Model 2: {self.model_2}")
        content.append(f"Mode: {self.mode}")
        if self.mode in ["sliding", "sliding_cache"]:
            content.append(f"Window Size: {self.window_size}")
        content.append(f"AI-Aware Mode: {self.ai_aware_mode}")
        content.append("")
        
        # Metrics
        content.append("METRICS:")
        content.append(f"Turns Completed: {metrics.turns_completed}")
        content.append(f"Total Input Tokens: {metrics.total_input_tokens:,}")
        content.append(f"Total Output Tokens: {metrics.total_output_tokens:,}")
        content.append(f"Total Cost: ${metrics.total_cost:.4f}")
        avg_time = sum(metrics.response_times) / len(metrics.response_times) if metrics.response_times else 0
        content.append(f"Average Response Time: {avg_time:.1f}s")
        if metrics.context_size_per_turn:
            avg_context = sum(metrics.context_size_per_turn) / len(metrics.context_size_per_turn)
            content.append(f"Average Context Size: {avg_context:.0f} tokens")
        content.append("")
        content.append("CONVERSATION:")
        content.append("=" * 50)
        content.append("")
        
        for i, turn in enumerate(self.conversation_history, 1):
            model_name = "Model 1" if turn.speaker == self.model_1 else "Model 2"
            timestamp = time.strftime("%H:%M:%S", time.localtime(turn.timestamp))
            
            content.append(f"TURN {i} - {model_name} ({turn.speaker}) - {timestamp}")
            content.append("-" * 50)
            content.append(turn.message)
            content.append("")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        print(f"\nConversation saved to {filename}")
    
    def print_conversation_summary(self):
        """Print a summary of the conversation."""
        print("\n" + "="*60)
        print("CONVERSATION SUMMARY")
        print("="*60)
        
        for i, turn in enumerate(self.conversation_history, 1):
            model_name = "Model 1" if turn.speaker == self.model_1 else "Model 2"
            print(f"\n{model_name} ({turn.speaker}):")
            print(f"{turn.message}")
            print("-" * 40)

def run_comparison_experiment(api_key: str, prompt: str, turns: int = 15, 
                             openai_api_key: Optional[str] = None, 
                             deepseek_api_key: Optional[str] = None,
                             enable_quality_metrics: bool = False):
    """Run the same conversation with all 4 modes for comparison."""
    modes = [
        {"mode": "full", "label": "Full Context"},
        {"mode": "sliding", "window_size": 8, "label": "Sliding Window (8)"},
        {"mode": "cache", "label": "Prompt Caching"},
        {"mode": "sliding_cache", "window_size": 8, "label": "Sliding + Cache (8)"}
    ]
    
    results = {}
    
    print("="*70)
    print("RUNNING COMPARISON EXPERIMENT")
    if enable_quality_metrics:
        print("Quality metrics ENABLED - this will take longer but provide detailed analysis")
    print("="*70)
    
    for config in modes:
        print(f"\n{'='*20} {config['label']} {'='*20}")
        
        conversation = LLMConversation(
            api_key=api_key,
            mode=config["mode"],
            window_size=config.get("window_size", 10),
            ai_aware_mode=False,
            openai_api_key=openai_api_key,
            deepseek_api_key=deepseek_api_key,
            enable_quality_metrics=enable_quality_metrics
        )
        
        metrics = conversation.start_conversation(
            initial_prompt=prompt,
            max_turns=turns,
            delay_between_turns=0.5,
            real_time_display=True
        )
        
        results[config["mode"]] = metrics
        
        # Save individual conversation
        filename = f"conversation_{config['mode']}.md"
        conversation.save_conversation(filename, metrics, "markdown")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"{'Mode':<20} {'Input Tokens':<12} {'Output Tokens':<13} {'Total Cost':<12} {'Avg Context':<12}")
    print("-" * 70)
    
    for mode, metrics in results.items():
        avg_context = sum(metrics.context_size_per_turn) / len(metrics.context_size_per_turn) if metrics.context_size_per_turn else 0
        print(f"{mode:<20} {metrics.total_input_tokens:<12,} {metrics.total_output_tokens:<13,} ${metrics.total_cost:<11.4f} {avg_context:<12.0f}")
    
    # Calculate savings
    full_cost = results["full"].total_cost
    print(f"\nCost Savings vs Full Context:")
    for mode, metrics in results.items():
        if mode != "full":
            savings = (full_cost - metrics.total_cost) / full_cost * 100
            print(f"{mode}: {savings:.1f}% savings")
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-to-AI Conversation Simulator')
    parser.add_argument('--mode', 
                       choices=['full', 'sliding', 'cache', 'sliding_cache'], 
                       default='full', 
                       help='Conversation mode (default: full)')
    parser.add_argument('--window-size', 
                       type=int, 
                       default=10, 
                       help='Sliding window size (default: 10)')
    parser.add_argument('--turns', 
                       type=int, 
                       default=20, 
                       help='Number of conversation turns (default: 20)')
    parser.add_argument('--ai-aware', 
                       action='store_true', 
                       help='Enable AI-aware mode')
    parser.add_argument('--compare', 
                       action='store_true', 
                       help='Run comparison of all 4 modes')
    parser.add_argument('--prompt', 
                       type=str, 
                       default="What's the most efficient way to solve complex optimization problems?",
                       help='Initial conversation prompt')
    parser.add_argument('--delay', 
                       type=float, 
                       default=0.5, 
                       help='Delay between turns in seconds (default: 0.5)')
    
    # Model selection
    parser.add_argument('--model-1', 
                       type=str, 
                       default='claude-3-sonnet-20240229',
                       help='First model (default: claude-3-sonnet-20240229)')
    parser.add_argument('--model-2', 
                       type=str, 
                       default='claude-3-sonnet-20240229',
                       help='Second model (default: claude-3-sonnet-20240229)')
    
    # API keys
    parser.add_argument('--anthropic-key', 
                       type=str, 
                       help='Anthropic API key (can also use env var ANTHROPIC_API_KEY)')
    parser.add_argument('--openai-key', 
                       type=str, 
                       help='OpenAI API key (can also use env var OPENAI_API_KEY)')
    parser.add_argument('--deepseek-key', 
                       type=str, 
                       help='DeepSeek API key (can also use env var DEEPSEEK_API_KEY)')
    # Quality metrics
    parser.add_argument('--quality-metrics', 
                       action='store_true', 
                       help='Enable conversation quality analysis with embeddings')
    parser.add_argument('--db-stats', 
                       action='store_true', 
                       help='Show embedding database statistics')
    
    # Template selection
    parser.add_argument('--template', 
                       choices=['debate', 'collaboration', 'socratic', 'creative', 'technical', 'learning'], 
                       help='Conversation template to use')
    parser.add_argument('--list-templates', 
                       action='store_true', 
                       help='List all available templates with descriptions')
    parser.add_argument('--template-prompt', 
                       action='store_true', 
                       help='Use a random prompt from the selected template')
    
    parser.add_argument('--save-format', 
                       choices=['json', 'markdown', 'txt'], 
                       default='markdown', 
                       help='Output format for saved conversations (default: markdown)')
    
    return parser.parse_args()

# Example usage
def main():
    import os
    
    # Load environment variables from .env file if available
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    args = parse_args()
    
    # Handle template listing
    if args.list_templates:
        template_manager = TemplateManager()
        print(template_manager.list_all_templates())
        return
    
    # Handle database stats
    if args.db_stats:
        db = EmbeddingDatabase()
        stats = db.get_database_stats()
        print("EMBEDDING DATABASE STATISTICS")
        print("=" * 40)
        print(f"Embeddings stored: {stats['embeddings_stored']:,}")
        print(f"Models used: {stats['models_used']}")
        print(f"Conversations tracked: {stats['conversations_tracked']}")
        print(f"Database size: {stats['database_size_mb']:.2f} MB")
        print(f"Database location: embeddings/cache.db")
        return
    
    # Get API keys from args or environment variables
    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY") or "your-anthropic-key-here"
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    deepseek_key = args.deepseek_key or os.getenv("DEEPSEEK_API_KEY")
    
    # Handle template prompt selection
    prompt = args.prompt
    if args.template and args.template_prompt:
        template_manager = TemplateManager()
        prompt = template_manager.get_random_prompt(args.template)
        print(f"Using template prompt: {prompt}")
    
    # Show template info if template is selected
    if args.template:
        template_manager = TemplateManager()
        print("\nTemplate Info:")
        print(template_manager.get_template_info(args.template))
        print()
    
    # Show quality metrics info if enabled
    if args.quality_metrics:
        print("\nQuality Metrics: ENABLED")
        print("This will analyze conversation using sentence embeddings.")
        print("First run will download ~80MB model to embeddings/models/")
        print("Embeddings will be cached in embeddings/cache.db")
        print()
    
    if args.compare:
        # Run comparison experiment
        results = run_comparison_experiment(
            api_key=anthropic_key,
            openai_api_key=openai_key,
            deepseek_api_key=deepseek_key,
            prompt=prompt,
            turns=args.turns,
            enable_quality_metrics=args.quality_metrics
        )
    else:
        # Run single conversation
        conversation = LLMConversation(
            api_key=anthropic_key,
            openai_api_key=openai_key,
            deepseek_api_key=deepseek_key,
            model_1=args.model_1,
            model_2=args.model_2,
            ai_aware_mode=args.ai_aware,
            mode=args.mode,
            window_size=args.window_size,
            template=args.template,
            enable_quality_metrics=args.quality_metrics
        )
        
        metrics = conversation.start_conversation(
            initial_prompt=prompt,
            max_turns=args.turns,
            delay_between_turns=args.delay,
            real_time_display=True
        )
        
        # Print summary and save
        conversation.print_conversation_summary()
        template_suffix = f"_{args.template}" if args.template else ""
        quality_suffix = "_quality" if args.quality_metrics else ""
        filename = f"conversation_{args.mode}{template_suffix}{quality_suffix}_{int(time.time())}.{args.save_format}"
        if args.save_format == "markdown":
            filename = filename.replace(".markdown", ".md")
        conversation.save_conversation(filename, metrics, args.save_format)

if __name__ == "__main__":
    main()
