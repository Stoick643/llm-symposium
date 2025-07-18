"""
Configuration Manager for LLM Symposium

Handles loading and managing configuration from JSON files and CLI arguments.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ConversationConfig:
    """Configuration for a conversation."""
    models: List[str]
    mode: str = "full"
    window_size: int = 10
    template: Optional[str] = None
    turns: int = 20
    ai_aware_mode: bool = False
    delay_between_turns: float = 0.5
    save_format: str = "markdown"
    enable_quality_metrics: bool = False
    initial_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationConfig':
        """Create config from dictionary."""
        # Handle legacy model_1/model_2 format
        if 'model_1' in data and 'model_2' in data:
            models = [data['model_1'], data['model_2']]
            data['models'] = models
            data.pop('model_1', None)
            data.pop('model_2', None)
        
        # Filter out unknown keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)


class ConfigManager:
    """Manages configuration loading and merging."""
    
    DEFAULT_CONFIG_PATH = "config.json"
    
    def __init__(self):
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> ConversationConfig:
        """Load default configuration."""
        return ConversationConfig(
            models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
            mode="full",
            window_size=10,
            template=None,
            turns=20,
            ai_aware_mode=False,
            delay_between_turns=0.5,
            save_format="markdown",
            enable_quality_metrics=False,
            initial_prompt=None
        )
    
    def load_config(self, config_path: Optional[str] = None) -> ConversationConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file. If None, looks for default config.json
            
        Returns:
            ConversationConfig object
        """
        # Try to load from specified path or default
        if config_path:
            return self._load_from_file(config_path)
        elif os.path.exists(self.DEFAULT_CONFIG_PATH):
            return self._load_from_file(self.DEFAULT_CONFIG_PATH)
        else:
            return self._load_default_config()
    
    def _load_from_file(self, config_path: str) -> ConversationConfig:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ConversationConfig.from_dict(data)
            
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration")
            return self._load_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file {config_path}: {e}")
            print("Using default configuration")
            return self._load_default_config()
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
            print("Using default configuration")
            return self._load_default_config()
    
    def merge_cli_args(self, config: ConversationConfig, cli_args: Dict[str, Any]) -> ConversationConfig:
        """
        Merge CLI arguments with config, CLI args take precedence.
        
        Args:
            config: Base configuration
            cli_args: CLI arguments to override config values
            
        Returns:
            Merged configuration
        """
        # Convert config to dict for easy manipulation
        config_dict = config.to_dict()
        
        # Apply CLI overrides
        for key, value in cli_args.items():
            if value is not None:  # Only override if CLI arg was provided
                if key in config_dict:
                    config_dict[key] = value
        
        # Handle special cases
        if 'models' in cli_args and cli_args['models']:
            # Parse comma-separated models string
            if isinstance(cli_args['models'], str):
                config_dict['models'] = [m.strip() for m in cli_args['models'].split(',')]
            else:
                config_dict['models'] = cli_args['models']
        
        return ConversationConfig.from_dict(config_dict)
    
    def save_config(self, config: ConversationConfig, config_path: str):
        """Save configuration to JSON file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
    
    def create_default_config_file(self):
        """Create a default config.json file."""
        default_config = ConversationConfig(
            models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
            mode="full",
            window_size=10,
            template=None,
            turns=20,
            ai_aware_mode=False,
            delay_between_turns=0.5,
            save_format="markdown",
            enable_quality_metrics=False,
            initial_prompt="What's the most efficient way to solve complex optimization problems?"
        )
        
        self.save_config(default_config, self.DEFAULT_CONFIG_PATH)
    
    def list_example_configs(self) -> List[str]:
        """List available example configuration files."""
        configs_dir = "configs"
        if not os.path.exists(configs_dir):
            return []
        
        config_files = []
        for filename in os.listdir(configs_dir):
            if filename.endswith('.json'):
                config_files.append(os.path.join(configs_dir, filename))
        
        return sorted(config_files)
    
    def validate_config(self, config: ConversationConfig) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate models
        if not config.models or len(config.models) < 2:
            errors.append("At least 2 models are required")
        
        # Validate mode
        valid_modes = ["full", "sliding", "cache", "sliding_cache"]
        if config.mode not in valid_modes:
            errors.append(f"Invalid mode: {config.mode}. Must be one of: {valid_modes}")
        
        # Validate window_size
        if config.window_size < 1:
            errors.append("Window size must be at least 1")
        
        # Validate turns
        if config.turns < 1:
            errors.append("Number of turns must be at least 1")
        
        # Validate delay
        if config.delay_between_turns < 0:
            errors.append("Delay between turns cannot be negative")
        
        # Validate save_format
        valid_formats = ["json", "markdown", "txt"]
        if config.save_format not in valid_formats:
            errors.append(f"Invalid save format: {config.save_format}. Must be one of: {valid_formats}")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


def get_api_keys():
    """Get API keys from environment variables."""
    return {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY')
    }