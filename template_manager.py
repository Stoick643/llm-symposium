"""
Template Manager for LLM Conversation Templates

Manages conversation templates for different interaction types like debates,
collaboration, socratic dialogue, creative writing, technical discussions, and learning.
"""

import random
from typing import List, Dict


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