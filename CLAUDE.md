# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your API keys:
```bash
ANTHROPIC_API_KEY=your-anthropic-key-here
OPENAI_API_KEY=your-openai-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
```

### Running the Application
```bash
python lll-talk.py [options]
```

### Key Command Line Options
- `--mode`: Set conversation mode (full, sliding, cache, sliding_cache)
- `--turns`: Number of conversation turns (default: 20)
- `--model-1` / `--model-2`: Specify models to use (Anthropic, OpenAI, or DeepSeek)
- `--template`: Use predefined conversation templates (debate, collaboration, socratic, creative, technical, learning)
- `--ai-aware`: Enable AI-aware mode where models know they're talking to another AI
- `--compare`: Run comparison experiment across all 4 modes
- `--quality-metrics`: Enable conversation quality analysis with embeddings
- `--save-format`: Output format (json, markdown, txt)

### Example Usage
```bash
# Basic conversation
python lll-talk.py --turns 15 --template debate

# Comparison experiment
python lll-talk.py --compare --turns 10 --template socratic

# Quality analysis
python lll-talk.py --quality-metrics --template technical
```

## Architecture

This is a single-file Python application (`lll-talk.py`) that orchestrates conversations between multiple AI models. The architecture consists of:

### Core Components

1. **TemplateManager**: Manages conversation templates with predefined system prompts and topics
   - 6 conversation types: debate, collaboration, socratic, creative, technical, learning
   - Each template has specific system prompts, suggested turn counts, and initial prompts

2. **LLMConversation**: Main conversation orchestrator
   - Supports multiple providers: Anthropic (Claude), OpenAI (GPT), DeepSeek
   - Implements 4 conversation modes:
     - `full`: Complete conversation history
     - `sliding`: Sliding window context management
     - `cache`: Prompt caching for efficiency
     - `sliding_cache`: Combined sliding window + caching

3. **Context Management**: Sophisticated context handling
   - Dynamic context building based on conversation mode
   - Token counting and cost tracking
   - Response time monitoring

4. **Quality Analysis** (optional): Conversation quality metrics using embeddings
   - Coherence scoring
   - Topic drift detection
   - Repetition analysis

### Key Features
- Multi-provider AI model support with automatic routing
- Real-time conversation display with metrics
- Comprehensive conversation export (JSON, Markdown, TXT)
- Cost tracking with provider-specific pricing
- Template-based conversation structuring
- AI-aware mode for AI-to-AI optimized communication

### Data Flow
1. User provides initial prompt and configuration
2. Conversation alternates between two selected models
3. Context is managed according to selected mode
4. Each turn is tracked with metrics (tokens, cost, timing)
5. Optional quality analysis runs on each message
6. Final conversation and metrics are saved to file

The application is designed for analyzing AI-to-AI conversations with different context management strategies and conversation templates.

## Web Dashboard

This project includes a comprehensive Flask web dashboard for real-time conversation viewing, historical replay, and AI-generated analysis. The dashboard provides:

- **Real-time conversation streaming** via WebSockets
- **Historical conversation replay** with playback controls
- **AI-generated summaries and keywords** for searchability
- **Text-to-speech integration** for audio consumption
- **Mobile-responsive design** for cross-device usage
- **Database persistence** with SQLite backend

See `DASHBOARD_SPECS.md` for detailed architecture, implementation specifications, and development setup instructions.