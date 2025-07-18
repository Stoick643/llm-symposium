"""
Tests for TemplateManager class.
"""

import pytest
from template_manager import TemplateManager


class TestTemplateManager:
    """Test cases for TemplateManager."""
    
    def test_init(self, template_manager):
        """Test TemplateManager initialization."""
        assert template_manager.templates is not None
        assert len(template_manager.templates) == 6
        assert "debate" in template_manager.templates
        assert "collaboration" in template_manager.templates
        assert "socratic" in template_manager.templates
        assert "creative" in template_manager.templates
        assert "technical" in template_manager.templates
        assert "learning" in template_manager.templates
    
    def test_list_templates(self, template_manager):
        """Test listing all template names."""
        templates = template_manager.list_templates()
        assert isinstance(templates, list)
        assert len(templates) == 6
        assert "debate" in templates
        assert "collaboration" in templates
    
    def test_get_template_valid(self, template_manager):
        """Test getting a valid template."""
        template = template_manager.get_template("debate")
        assert isinstance(template, dict)
        assert "name" in template
        assert "description" in template
        assert "system_prompt" in template
        assert "initial_prompts" in template
        assert "suggested_turns" in template
        assert "mode_recommendation" in template
        assert template["name"] == "Formal Debate"
    
    def test_get_template_invalid(self, template_manager):
        """Test getting an invalid template raises ValueError."""
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            template_manager.get_template("invalid")
    
    def test_get_system_prompt(self, template_manager):
        """Test getting system prompt for a template."""
        system_prompt = template_manager.get_system_prompt("debate")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
        assert "debate" in system_prompt.lower()
    
    def test_get_random_prompt(self, template_manager):
        """Test getting a random prompt from a template."""
        prompt = template_manager.get_random_prompt("debate")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Check that it's one of the initial prompts
        template = template_manager.get_template("debate")
        assert prompt in template["initial_prompts"]
    
    def test_get_template_info(self, template_manager):
        """Test getting formatted template information."""
        info = template_manager.get_template_info("debate")
        assert isinstance(info, str)
        assert "Formal Debate" in info
        assert "Description:" in info
        assert "Suggested turns:" in info
        assert "Recommended mode:" in info
        assert "Available prompts:" in info
    
    def test_list_all_templates(self, template_manager):
        """Test getting formatted list of all templates."""
        all_templates = template_manager.list_all_templates()
        assert isinstance(all_templates, str)
        assert "Available Conversation Templates:" in all_templates
        assert "DEBATE:" in all_templates
        assert "COLLABORATION:" in all_templates
        assert "SOCRATIC:" in all_templates
        assert "CREATIVE:" in all_templates
        assert "TECHNICAL:" in all_templates
        assert "LEARNING:" in all_templates
    
    def test_template_structure(self, template_manager):
        """Test that all templates have the required structure."""
        required_keys = ["name", "description", "system_prompt", "initial_prompts", "suggested_turns", "mode_recommendation"]
        
        for template_name in template_manager.list_templates():
            template = template_manager.get_template(template_name)
            
            # Check all required keys are present
            for key in required_keys:
                assert key in template, f"Template '{template_name}' missing key '{key}'"
            
            # Check data types
            assert isinstance(template["name"], str)
            assert isinstance(template["description"], str)
            assert isinstance(template["system_prompt"], str)
            assert isinstance(template["initial_prompts"], list)
            assert isinstance(template["suggested_turns"], int)
            assert isinstance(template["mode_recommendation"], str)
            
            # Check non-empty values
            assert len(template["name"]) > 0
            assert len(template["description"]) > 0
            assert len(template["system_prompt"]) > 0
            assert len(template["initial_prompts"]) > 0
            assert template["suggested_turns"] > 0
            assert template["mode_recommendation"] in ["full", "sliding", "cache", "sliding_cache"]
    
    def test_initial_prompts_quality(self, template_manager):
        """Test that initial prompts are meaningful."""
        for template_name in template_manager.list_templates():
            template = template_manager.get_template(template_name)
            
            for prompt in template["initial_prompts"]:
                assert isinstance(prompt, str)
                assert len(prompt) > 20  # Reasonably long prompts
                assert "?" in prompt or prompt.endswith(".")  # Proper punctuation