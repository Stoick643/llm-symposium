"""
Conversation Manager for LLM-to-LLM Conversations

Handles the core conversation logic between two LLM models with support for
different conversation modes, context management, and metrics tracking.
"""

import anthropic
from openai import OpenAI
import time
import json
from typing import List, Dict, Optional

from models import ConversationTurn, ConversationMetrics, model_manager, Provider
from template_manager import TemplateManager
from config_manager import ConversationConfig


class LLMConversation:
    """Manages conversations between multiple LLM models."""
    
    def __init__(self, config: ConversationConfig, 
                 api_key: str,
                 openai_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None,
                 moonshot_api_key: Optional[str] = None):
        """
        Initialize the conversation between multiple LLM models.
        
        Args:
            config: ConversationConfig object with all settings
            api_key: Anthropic API key (also used as default for others if not specified)
            openai_api_key: OpenAI API key (if None, uses api_key)
            deepseek_api_key: DeepSeek API key (if None, uses api_key)
            moonshot_api_key: Moonshot API key (if None, uses api_key)
        """
        self.config = config
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        self.openai_client = OpenAI(api_key=openai_api_key or api_key)
        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key or api_key,
            base_url="https://api.deepseek.com"
        )
        self.moonshot_client = OpenAI(
            api_key=moonshot_api_key or api_key,
            base_url="https://api.moonshot.ai/v1"
        )
        
        # Multi-model setup
        self.models = config.models
        self.current_model_index = 0
        self.conversation_history: List[ConversationTurn] = []
        
        # Template management
        self.template_manager = TemplateManager()
        self.template_config = None
        if config.template:
            self.template_config = self.template_manager.get_template(config.template)
        
        # Quality analysis - placeholder for now
        self.quality_analyzer = None  # Will implement later
        
        # Metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.context_size_per_turn = []
        self.response_times = []
        
    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Get pricing for a specific model."""
        pricing = model_manager.get_pricing(model)
        if pricing:
            return pricing
        # Default to Claude pricing if model not found
        return {"input": 3.0, "output": 15.0}
    
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
            if self.config.mode in ["cache", "sliding_cache"]:
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
        if self.template_config and not self.config.ai_aware_mode:
            template_system_prompt = self.template_config["system_prompt"]
            messages.insert(0, {"role": "system", "content": template_system_prompt})
        
        # Add AI-aware system prompt if enabled (overrides template)
        if self.config.ai_aware_mode:
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
        
        # Route to appropriate provider using model manager
        provider = model_manager.get_provider_for_model(model)
        if provider == Provider.ANTHROPIC:
            return self._call_anthropic_model(model, messages)
        elif provider == Provider.OPENAI:
            return self._call_openai_compatible_model(model, messages, self.openai_client)
        elif provider == Provider.DEEPSEEK:
            return self._call_openai_compatible_model(model, messages, self.deepseek_client)
        elif provider == Provider.MOONSHOT:
            return self._call_openai_compatible_model(model, messages, self.moonshot_client)
        else:
            # Unknown model - try Anthropic as default
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
        if len(full_context) <= self.config.window_size:
            return full_context
        
        # Keep first 2 messages (initial context) + last (window_size - 2) messages
        if self.config.window_size >= 2:
            return full_context[:2] + full_context[-(self.config.window_size - 2):]
        else:
            return full_context[-self.config.window_size:]
    
    def _build_context_for_model(self, current_model: str) -> List[Dict]:
        """Build conversation context for a model, applying the selected mode."""
        # Get full context
        full_context = self._build_full_context(current_model)
        
        # Apply sliding window if enabled
        if self.config.mode in ["sliding", "sliding_cache"]:
            context = self._apply_sliding_window(full_context)
        else:
            context = full_context
        
        return context
    
    def start_conversation(self, initial_prompt: str, max_turns: int = 10, 
                          delay_between_turns: float = 0.5, 
                          real_time_display: bool = True) -> ConversationMetrics:
        """
        Start a conversation between multiple models.
        
        Args:
            initial_prompt: The topic/question to start the conversation
            max_turns: Maximum number of conversation turns
            delay_between_turns: Delay in seconds between API calls
            real_time_display: Whether to display conversation in real-time
            
        Returns:
            ConversationMetrics object with detailed statistics
        """
        print(f"Starting conversation with initial prompt: {initial_prompt}")
        for i, model in enumerate(self.models, 1):
            print(f"Model {i}: {model}")
        print(f"Mode: {self.config.mode}")
        if self.config.mode in ["sliding", "sliding_cache"]:
            print(f"Window Size: {self.config.window_size}")
        print(f"AI-Aware Mode: {self.config.ai_aware_mode}")
        if self.config.template:
            print(f"Template: {self.config.template} ({self.template_config['name']})")
        if self.config.enable_quality_metrics:
            print(f"Quality Metrics: Enabled (using embeddings)")
        print("-" * 60)
        
        # Generate unique conversation ID for database tracking
        conversation_id = f"conv_{int(time.time())}_{hash(initial_prompt) % 10000}"
        
        # First model starts the conversation
        current_model = self.models[self.current_model_index]
        current_prompt = initial_prompt
        
        for turn in range(max_turns):
            if real_time_display:
                model_name = f"Model {self.current_model_index + 1}"
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
            
            # Analyze message quality if enabled (placeholder)
            if self.config.enable_quality_metrics and self.quality_analyzer:
                # Quality analysis would go here
                pass
            
            if real_time_display:
                print(f"\r{response}")
                print(f"(Tokens: {input_tokens} in, {output_tokens} out | Time: {response_time:.1f}s)")
                if self.config.mode in ["sliding", "sliding_cache"]:
                    print(f"(Context size: {len(context)} messages)")
            
            # Switch to the next model (circular)
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            current_model = self.models[self.current_model_index]
            
            # The next prompt is the previous response
            current_prompt = response
            
            # Optional delay (can be set to 0 for maximum speed)
            if delay_between_turns > 0:
                time.sleep(delay_between_turns)
        
        # Calculate total cost using model-specific pricing
        model_pricings = [self._get_pricing(model) for model in self.models]
        
        # Rough estimate: split tokens evenly between models
        avg_input_cost = sum(pricing["input"] for pricing in model_pricings) / len(model_pricings)
        avg_output_cost = sum(pricing["output"] for pricing in model_pricings) / len(model_pricings)
        
        total_cost = (
            (self.total_input_tokens / 1_000_000) * avg_input_cost +
            (self.total_output_tokens / 1_000_000) * avg_output_cost
        )
        
        # Create metrics object
        metrics = ConversationMetrics(
            mode=self.config.mode,
            window_size=self.config.window_size if self.config.mode in ["sliding", "sliding_cache"] else None,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost=total_cost,
            context_size_per_turn=self.context_size_per_turn,
            response_times=self.response_times,
            turns_completed=len(self.conversation_history)
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
    
    def print_conversation_summary(self):
        """Print a summary of the conversation."""
        print("\n" + "="*60)
        print("CONVERSATION SUMMARY")
        print("="*60)
        
        for i, turn in enumerate(self.conversation_history, 1):
            try:
                model_index = self.models.index(turn.speaker)
                model_name = f"Model {model_index + 1}"
            except ValueError:
                model_name = "Unknown Model"
            print(f"\n{model_name} ({turn.speaker}):")
            print(f"{turn.message}")
            print("-" * 40)