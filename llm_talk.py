#!/usr/bin/env python3
"""
LLM Talk - AI-to-AI Conversation Simulator

Main CLI entry point for running conversations between LLM models.
"""

import os
import time
import json
import argparse
from typing import List, Dict, Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from conversation_manager import LLMConversation
from template_manager import TemplateManager
from models import ConversationMetrics
from config_manager import ConfigManager, ConversationConfig


def run_comparison_experiment(api_key: str, prompt: str, turns: int = 15, 
                             openai_api_key: Optional[str] = None, 
                             deepseek_api_key: Optional[str] = None,
                             moonshot_api_key: Optional[str] = None,
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
        
        # Create config object for this mode
        conv_config = ConversationConfig(
            models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
            mode=config["mode"],
            window_size=config.get("window_size", 10),
            ai_aware_mode=False,
            turns=turns,
            enable_quality_metrics=enable_quality_metrics
        )
        
        conversation = LLMConversation(
            config=conv_config,
            api_key=api_key,
            openai_api_key=openai_api_key,
            deepseek_api_key=deepseek_api_key,
            moonshot_api_key=moonshot_api_key
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
        save_conversation_markdown(conversation, metrics, filename)
    
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


def save_conversation_markdown(conversation: LLMConversation, metrics: ConversationMetrics, filename: str):
    """Save conversation in Markdown format."""
    content = []
    
    # Header
    content.append("# AI-to-AI Conversation")
    content.append("")
    content.append("## Configuration")
    for i, model in enumerate(conversation.models, 1):
        content.append(f"- **Model {i}**: {model}")
    content.append(f"- **Mode**: {conversation.config.mode}")
    if conversation.config.mode in ["sliding", "sliding_cache"]:
        content.append(f"- **Window Size**: {conversation.config.window_size}")
    content.append(f"- **AI-Aware Mode**: {conversation.config.ai_aware_mode}")
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
    
    for i, turn in enumerate(conversation.conversation_history, 1):
        try:
            model_index = conversation.models.index(turn.speaker)
            model_name = f"Model {model_index + 1}"
        except ValueError:
            model_name = "Unknown Model"
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-to-AI Conversation Simulator')
    
    # Config file option
    parser.add_argument('--config', 
                       type=str, 
                       help='Path to JSON config file (default: config.json if exists)')
    
    # Basic settings (can override config file)
    parser.add_argument('--mode', 
                       choices=['full', 'sliding', 'cache', 'sliding_cache'], 
                       help='Conversation mode (overrides config)')
    parser.add_argument('--window-size', 
                       type=int, 
                       help='Sliding window size (overrides config)')
    parser.add_argument('--turns', 
                       type=int, 
                       help='Number of conversation turns (overrides config)')
    parser.add_argument('--ai-aware', 
                       action='store_true', 
                       help='Enable AI-aware mode (overrides config)')
    parser.add_argument('--compare', 
                       action='store_true', 
                       help='Run comparison of all 4 modes')
    parser.add_argument('--prompt', 
                       type=str, 
                       help='Initial conversation prompt (overrides config)')
    parser.add_argument('--delay', 
                       type=float, 
                       help='Delay between turns in seconds (overrides config)')
    
    # Model selection (backward compatibility)
    parser.add_argument('--model-1', 
                       type=str, 
                       help='First model (overrides config)')
    parser.add_argument('--model-2', 
                       type=str, 
                       help='Second model (overrides config)')
    parser.add_argument('--models', 
                       type=str, 
                       nargs='+',
                       help='List of models to use (overrides config)')
    
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
    parser.add_argument('--moonshot-key', 
                       type=str, 
                       help='Moonshot API key (can also use env var MOONSHOT_API_KEY)')
    
    # Quality metrics
    parser.add_argument('--quality-metrics', 
                       action='store_true', 
                       help='Enable conversation quality analysis with embeddings')
    
    # Database options
    parser.add_argument('--save-to-db', 
                       action='store_true', 
                       help='Save conversation to SQLite database')
    
    # Template selection
    parser.add_argument('--template', 
                       choices=['debate', 'collaboration', 'socratic', 'creative', 'technical', 'learning'], 
                       help='Conversation template to use (overrides config)')
    parser.add_argument('--list-templates', 
                       action='store_true', 
                       help='List all available templates with descriptions')
    parser.add_argument('--list-models', 
                       action='store_true', 
                       help='List all available models with descriptions')
    parser.add_argument('--template-prompt', 
                       action='store_true', 
                       help='Use a random prompt from the selected template')
    
    parser.add_argument('--save-format', 
                       choices=['json', 'markdown', 'txt'], 
                       help='Output format for saved conversations (overrides config)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Load environment variables from .env file if available
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    args = parse_args()
    
    # Handle template listing
    if args.list_templates:
        template_manager = TemplateManager()
        print(template_manager.list_all_templates())
        return
    
    # Handle model listing
    if args.list_models:
        from models import model_manager
        print(model_manager.list_models_formatted())
        return
    
    # Load configuration with CLI overrides
    config_manager = ConfigManager()
    config_path = args.config
    
    # Auto-detect config file if not specified
    if not config_path and os.path.exists("config.json"):
        config_path = "config.json"
        print(f"Auto-detected config file: {config_path}")
    
    # Load config
    if config_path:
        config = config_manager.load_config(config_path)
        print(f"Loaded config from: {config_path}")
    else:
        # Use default config
        config = ConversationConfig()
        print("Using default configuration")
    
    # Apply CLI overrides
    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.window_size:
        overrides["window_size"] = args.window_size
    if args.turns:
        overrides["turns"] = args.turns
    if args.ai_aware:
        overrides["ai_aware_mode"] = args.ai_aware
    if args.prompt:
        overrides["initial_prompt"] = args.prompt
    if args.delay:
        overrides["delay_between_turns"] = args.delay
    if args.template:
        overrides["template"] = args.template
    if args.save_format:
        overrides["save_format"] = args.save_format
    if args.quality_metrics:
        overrides["enable_quality_metrics"] = args.quality_metrics
    
    # Handle model overrides
    if args.models:
        overrides["models"] = args.models
    elif args.model_1 or args.model_2:
        # Legacy support for --model-1 and --model-2
        models = [args.model_1 or config.models[0], args.model_2 or (config.models[1] if len(config.models) > 1 else config.models[0])]
        overrides["models"] = models
    
    # Apply overrides
    if overrides:
        config = config_manager.merge_cli_args(config, overrides)
        print(f"Applied CLI overrides: {', '.join(overrides.keys())}")
    
    # Get API keys from args or environment variables
    anthropic_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY") or "your-anthropic-key-here"
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    deepseek_key = args.deepseek_key or os.getenv("DEEPSEEK_API_KEY")
    moonshot_key = args.moonshot_key or os.getenv("MOONSHOT_API_KEY")
    
    # Handle template prompt selection
    prompt = config.initial_prompt
    if args.template and args.template_prompt:
        template_manager = TemplateManager()
        prompt = template_manager.get_random_prompt(args.template)
        print(f"Using template prompt: {prompt}")
    
    # Show template info if template is selected
    if config.template:
        template_manager = TemplateManager()
        print("\nTemplate Info:")
        print(template_manager.get_template_info(config.template))
        print()
    
    # Show quality metrics info if enabled
    if config.enable_quality_metrics:
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
            moonshot_api_key=moonshot_key,
            prompt=prompt,
            turns=config.turns,
            enable_quality_metrics=config.enable_quality_metrics
        )
    else:
        # Run single conversation
        conversation = LLMConversation(
            config=config,
            api_key=anthropic_key,
            openai_api_key=openai_key,
            deepseek_api_key=deepseek_key,
            moonshot_api_key=moonshot_key,
            save_to_db=args.save_to_db
        )
        
        metrics = conversation.start_conversation(
            initial_prompt=prompt,
            max_turns=config.turns,
            delay_between_turns=config.delay_between_turns,
            real_time_display=True
        )
        
        # Print summary and save
        conversation.print_conversation_summary()
        template_suffix = f"_{config.template}" if config.template else ""
        quality_suffix = "_quality" if config.enable_quality_metrics else ""
        filename = f"conversation_{config.mode}{template_suffix}{quality_suffix}_{int(time.time())}.{config.save_format}"
        if config.save_format == "markdown":
            filename = filename.replace(".markdown", ".md")
            save_conversation_markdown(conversation, metrics, filename)
        else:
            print(f"Note: Only markdown format is currently supported. Saved as {filename}")


if __name__ == "__main__":
    main()