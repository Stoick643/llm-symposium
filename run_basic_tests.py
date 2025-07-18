#!/usr/bin/env python3
"""
Simple test runner that doesn't require pytest.
"""

import sys
import traceback
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, '.')

def test_model_manager():
    """Test the model manager functionality."""
    print("🔧 Testing Model Manager...")
    
    try:
        from models import model_manager, Provider
        
        # Test 1: Check if model manager initializes
        assert model_manager is not None, "Model manager should be initialized"
        print("  ✅ Model manager initializes")
        
        # Test 2: Check if providers are loaded
        providers = model_manager.get_all_providers()
        assert len(providers) == 4, f"Expected 4 providers, got {len(providers)}"
        print("  ✅ All 4 providers loaded")
        
        # Test 3: Check if models are loaded
        models = model_manager.get_all_models()
        assert len(models) == 9, f"Expected 9 models, got {len(models)}"
        print("  ✅ All 9 models loaded")
        
        # Test 4: Check specific model (Moonshot)
        kimi_model = model_manager.get_model("kimi-k2-0711-preview")
        assert kimi_model is not None, "Kimi model should exist"
        assert kimi_model.provider == Provider.MOONSHOT, "Kimi should be Moonshot provider"
        print("  ✅ Moonshot Kimi model found")
        
        # Test 5: Check client config
        config = model_manager.get_client_config("kimi-k2-0711-preview")
        assert config is not None, "Client config should exist"
        assert config["base_url"] == "https://api.moonshot.ai/v1", "Base URL should be correct"
        print("  ✅ Client config correct")
        
        # Test 6: Check model validation
        assert model_manager.is_valid_model("kimi-k2-0711-preview"), "Kimi should be valid"
        assert not model_manager.is_valid_model("fake-model"), "Fake model should be invalid"
        print("  ✅ Model validation works")
        
        print("🎉 Model Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_config_manager():
    """Test the config manager functionality."""
    print("🔧 Testing Config Manager...")
    
    try:
        from config_manager import ConfigManager, ConversationConfig
        
        # Test 1: Check if config manager initializes
        config_manager = ConfigManager()
        assert config_manager is not None, "Config manager should initialize"
        print("  ✅ Config manager initializes")
        
        # Test 2: Check default config
        default_config = ConversationConfig(models=["claude-3-sonnet-20240229", "gpt-4"])
        assert default_config.mode == "full", "Default mode should be full"
        assert len(default_config.models) >= 2, "Should have at least 2 models"
        print("  ✅ Default config works")
        
        # Test 3: Check config loading
        config = config_manager.load_config("config.json")
        assert config is not None, "Config should load"
        assert "kimi-k2-0711-preview" in config.models, "Should have Kimi model"
        print("  ✅ Config loading works")
        
        print("🎉 Config Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Config Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_conversation_models():
    """Test the conversation data models."""
    print("🔧 Testing Conversation Models...")
    
    try:
        from models import ConversationTurn, ConversationMetrics
        
        # Test 1: ConversationTurn
        turn = ConversationTurn(
            speaker="test-model",
            message="Test message",
            timestamp=1234567890.0
        )
        assert turn.speaker == "test-model"
        assert turn.message == "Test message"
        print("  ✅ ConversationTurn works")
        
        # Test 2: ConversationMetrics
        metrics = ConversationMetrics(
            mode="full",
            window_size=10,
            total_input_tokens=100,
            total_output_tokens=200,
            total_cost=0.05,
            context_size_per_turn=[10, 20],
            response_times=[1.0, 2.0],
            turns_completed=2
        )
        assert metrics.mode == "full"
        assert metrics.total_input_tokens == 100
        print("  ✅ ConversationMetrics works")
        
        print("🎉 Conversation Models tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Conversation Models test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Running Basic Tests...")
    print("=" * 50)
    
    results = []
    results.append(test_model_manager())
    results.append(test_config_manager())
    results.append(test_conversation_models())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        return 0
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total})")
        return 1

if __name__ == "__main__":
    sys.exit(main())