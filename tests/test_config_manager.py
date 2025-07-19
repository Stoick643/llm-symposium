"""
Test suite for config_manager.py

Tests configuration loading, validation, CLI argument merging, and legacy support.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, mock_open

from config_manager import ConfigManager, ConversationConfig


class TestConversationConfig:
    """Test the ConversationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationConfig(models=["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"])
        
        assert config.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
        assert config.mode == "full"
        assert config.window_size == 10
        assert config.turns == 15
        assert config.ai_aware_mode is False
        assert config.template is None
        assert config.initial_prompt == "What is the most efficient way to sort a list of numbers?"
        assert config.delay_between_turns == 0.5
        assert config.save_format == "markdown"
        assert config.enable_quality_metrics is False
    
    def test_config_from_dict_complete(self):
        """Test creating config from complete dictionary."""
        data = {
            "models": ["gpt-4", "claude-3-sonnet-20240229"],
            "mode": "sliding",
            "window_size": 8,
            "turns": 20,
            "ai_aware_mode": True,
            "template": "debate",
            "initial_prompt": "Custom prompt",
            "delay_between_turns": 1.0,
            "save_format": "json",
            "enable_quality_metrics": True
        }
        
        config = ConversationConfig.from_dict(data)
        
        assert config.models == ["gpt-4", "claude-3-sonnet-20240229"]
        assert config.mode == "sliding"
        assert config.window_size == 8
        assert config.turns == 20
        assert config.ai_aware_mode is True
        assert config.template == "debate"
        assert config.initial_prompt == "Custom prompt"
        assert config.delay_between_turns == 1.0
        assert config.save_format == "json"
        assert config.enable_quality_metrics is True
    
    def test_config_from_dict_partial(self):
        """Test creating config from partial dictionary (uses defaults)."""
        data = {
            "models": ["gpt-4"],
            "mode": "cache",
            "turns": 10
        }
        
        config = ConversationConfig.from_dict(data)
        
        assert config.models == ["gpt-4"]
        assert config.mode == "cache"
        assert config.turns == 10
        # These should use defaults
        assert config.window_size == 10
        assert config.ai_aware_mode is False
        assert config.template is None
    
    def test_config_from_dict_legacy_format(self):
        """Test creating config from legacy format with model_1 and model_2."""
        data = {
            "model_1": "claude-3-sonnet-20240229",
            "model_2": "gpt-4",
            "mode": "full",
            "turns": 12
        }
        
        config = ConversationConfig.from_dict(data)
        
        # Should convert legacy format to new models list
        assert config.models == ["claude-3-sonnet-20240229", "gpt-4"]
        assert config.mode == "full"
        assert config.turns == 12
    
    def test_config_from_dict_legacy_with_models_override(self):
        """Test that new 'models' field takes precedence over legacy fields."""
        data = {
            "model_1": "old-model-1",
            "model_2": "old-model-2",
            "models": ["new-model-1", "new-model-2", "new-model-3"],
            "mode": "sliding"
        }
        
        config = ConversationConfig.from_dict(data)
        
        # New models field should override legacy
        assert config.models == ["new-model-1", "new-model-2", "new-model-3"]
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ConversationConfig(
            models=["test-model-1", "test-model-2"],
            mode="sliding_cache",
            window_size=6,
            turns=8,
            ai_aware_mode=True,
            template="creative"
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["models"] == ["test-model-1", "test-model-2"]
        assert config_dict["mode"] == "sliding_cache"
        assert config_dict["window_size"] == 6
        assert config_dict["turns"] == 8
        assert config_dict["ai_aware_mode"] is True
        assert config_dict["template"] == "creative"


class TestConfigManager:
    """Test the ConfigManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a ConfigManager instance."""
        return ConfigManager()
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config_data = {
            "models": ["claude-3-sonnet-20240229", "gpt-4"],
            "mode": "sliding",
            "window_size": 8,
            "turns": 12,
            "template": "debate"
        }
        
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(config_data, f)
        
        yield path
        os.unlink(path)
    
    def test_load_config_from_file_success(self, manager, temp_config_file):
        """Test successful config loading from file."""
        config = manager.load_config(temp_config_file)
        
        assert config.models == ["claude-3-sonnet-20240229", "gpt-4"]
        assert config.mode == "sliding"
        assert config.window_size == 8
        assert config.turns == 12
        assert config.template == "debate"
    
    def test_load_config_file_not_found(self, manager):
        """Test loading config from nonexistent file."""
        config = manager.load_config("nonexistent.json")
        
        # Should return default config
        assert config.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
        assert config.mode == "full"
    
    def test_load_config_invalid_json(self, manager):
        """Test loading config from file with invalid JSON."""
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            f.write('{"invalid": json,}')  # Invalid JSON
        
        try:
            config = manager.load_config(path)
            # Should return default config on error
            assert config.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
        finally:
            os.unlink(path)
    
    def test_save_config_success(self, manager):
        """Test successful config saving."""
        config = ConversationConfig(
            models=["test-model"],
            mode="cache",
            turns=5
        )
        
        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        
        try:
            success = manager.save_config(config, path)
            assert success is True
            
            # Verify file was written correctly
            with open(path, 'r') as f:
                data = json.load(f)
            
            assert data["models"] == ["test-model"]
            assert data["mode"] == "cache"
            assert data["turns"] == 5
            
        finally:
            os.unlink(path)
    
    def test_save_config_permission_error(self, manager):
        """Test config saving with permission error."""
        config = ConversationConfig()
        
        # Try to save to root directory (should fail)
        success = manager.save_config(config, "/root/config.json")
        assert success is False
    
    def test_merge_cli_args_complete(self, manager):
        """Test merging CLI arguments with existing config."""
        base_config = ConversationConfig(
            models=["base-model"],
            mode="full",
            turns=10
        )
        
        cli_overrides = {
            "mode": "sliding",
            "window_size": 6,
            "turns": 15,
            "ai_aware_mode": True,
            "template": "socratic"
        }
        
        merged = manager.merge_cli_args(base_config, cli_overrides)
        
        # CLI args should override
        assert merged.mode == "sliding"
        assert merged.window_size == 6
        assert merged.turns == 15
        assert merged.ai_aware_mode is True
        assert merged.template == "socratic"
        
        # Non-overridden values should remain
        assert merged.models == ["base-model"]
    
    def test_merge_cli_args_models_override(self, manager):
        """Test merging CLI arguments with models override."""
        base_config = ConversationConfig(
            models=["old-model-1", "old-model-2"]
        )
        
        cli_overrides = {
            "models": ["new-model-1", "new-model-2", "new-model-3"]
        }
        
        merged = manager.merge_cli_args(base_config, cli_overrides)
        
        assert merged.models == ["new-model-1", "new-model-2", "new-model-3"]
    
    def test_merge_cli_args_empty_overrides(self, manager):
        """Test merging with empty CLI overrides."""
        base_config = ConversationConfig(
            models=["test-model"],
            mode="cache",
            turns=8
        )
        
        merged = manager.merge_cli_args(base_config, {})
        
        # Should be identical to base config
        assert merged.models == ["test-model"]
        assert merged.mode == "cache"
        assert merged.turns == 8
    
    def test_merge_cli_args_none_values_ignored(self, manager):
        """Test that None values in CLI args are ignored."""
        base_config = ConversationConfig(
            models=["test-model"],
            mode="full",
            turns=10
        )
        
        cli_overrides = {
            "mode": None,
            "turns": 15,
            "template": None
        }
        
        merged = manager.merge_cli_args(base_config, cli_overrides)
        
        # None values should be ignored
        assert merged.mode == "full"  # Not overridden
        assert merged.turns == 15     # Overridden
        assert merged.template is None  # Original value
    
    def test_validate_config_valid(self, manager):
        """Test validation of valid configuration."""
        config = ConversationConfig(
            models=["claude-3-sonnet-20240229", "gpt-4"],
            mode="sliding",
            window_size=8,
            turns=10
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is True
    
    def test_validate_config_empty_models(self, manager):
        """Test validation with empty models list."""
        config = ConversationConfig(
            models=[],
            mode="full",
            turns=10
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_mode(self, manager):
        """Test validation with invalid mode."""
        config = ConversationConfig(
            models=["test-model"],
            mode="invalid_mode",
            turns=10
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_negative_turns(self, manager):
        """Test validation with negative turns."""
        config = ConversationConfig(
            models=["test-model"],
            mode="full",
            turns=-5
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_window_size(self, manager):
        """Test validation with invalid window size."""
        config = ConversationConfig(
            models=["test-model"],
            mode="sliding",
            window_size=0  # Should be > 0
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_save_format(self, manager):
        """Test validation with invalid save format."""
        config = ConversationConfig(
            models=["test-model"],
            save_format="invalid_format"
        )
        
        is_valid = manager.validate_config(config)
        assert is_valid is False


class TestConfigFileFormats:
    """Test various config file formats and edge cases."""
    
    @pytest.fixture
    def manager(self):
        return ConfigManager()
    
    def test_legacy_config_format(self, manager):
        """Test loading legacy config format."""
        legacy_data = {
            "model_1": "claude-3-sonnet-20240229",
            "model_2": "gpt-4",
            "mode": "full",
            "max_turns": 15,  # Old field name
            "window_size": 10
        }
        
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(legacy_data, f)
        
        try:
            config = manager.load_config(path)
            
            # Should convert to new format
            assert config.models == ["claude-3-sonnet-20240229", "gpt-4"]
            assert config.mode == "full"
            # max_turns should be converted to turns
            assert hasattr(config, 'turns')
            
        finally:
            os.unlink(path)
    
    def test_config_with_extra_fields(self, manager):
        """Test config with unknown fields (should be ignored)."""
        data = {
            "models": ["test-model"],
            "mode": "full",
            "turns": 10,
            "unknown_field": "value",
            "another_unknown": 123
        }
        
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        
        try:
            config = manager.load_config(path)
            
            # Known fields should be loaded
            assert config.models == ["test-model"]
            assert config.mode == "full"
            assert config.turns == 10
            
            # Unknown fields should be ignored (no error)
            
        finally:
            os.unlink(path)
    
    def test_empty_config_file(self, manager):
        """Test loading empty config file."""
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            f.write('{}')
        
        try:
            config = manager.load_config(path)
            
            # Should use defaults for all fields
            assert config.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
            assert config.mode == "full"
            
        finally:
            os.unlink(path)
    
    def test_config_with_comments(self, manager):
        """Test that JSON with comments fails gracefully."""
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            f.write('''
            {
                // This is a comment
                "models": ["test-model"],
                "mode": "full"  // Another comment
            }
            ''')
        
        try:
            config = manager.load_config(path)
            
            # Should fall back to defaults (JSON comments are invalid)
            assert config.models == ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"]
            
        finally:
            os.unlink(path)