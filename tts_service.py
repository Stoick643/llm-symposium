"""
Text-to-Speech Service using OpenAI TTS API

Provides text-to-speech functionality for LLM conversations with support for
multiple voices and AI model-specific voice mapping.
"""

import os
import tempfile
from typing import Optional, Dict, List
from pathlib import Path
from openai import OpenAI

from models import Provider, model_manager


class TTSService:
    """Text-to-Speech service using OpenAI TTS API."""
    
    # Available OpenAI TTS voices
    AVAILABLE_VOICES = [
        "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    ]
    
    # Default voice mapping for different providers/models
    DEFAULT_VOICE_MAPPING = {
        Provider.ANTHROPIC: "alloy",
        Provider.OPENAI: "nova", 
        Provider.DEEPSEEK: "echo",
        Provider.MOONSHOT: "fable"
    }
    
    def __init__(self, api_key: Optional[str] = None, voice_mapping: Optional[Dict] = None):
        """
        Initialize the TTS service.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            voice_mapping: Custom mapping of providers to voices
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required for TTS service")
        
        self.client = OpenAI(api_key=self.api_key)
        self.voice_mapping = voice_mapping or self.DEFAULT_VOICE_MAPPING.copy()
        
        # Ensure audio directory exists
        self.audio_dir = Path("audio_cache")
        self.audio_dir.mkdir(exist_ok=True)
    
    def get_voice_for_model(self, model_name: str) -> str:
        """
        Get the appropriate voice for a specific model.
        
        Args:
            model_name: Name of the AI model
            
        Returns:
            Voice name to use for TTS
        """
        provider = model_manager.get_provider_for_model(model_name)
        if provider and provider in self.voice_mapping:
            return self.voice_mapping[provider]
        
        # Default fallback voice
        return "alloy"
    
    def set_voice_for_provider(self, provider: Provider, voice: str) -> None:
        """
        Set a custom voice for a specific provider.
        
        Args:
            provider: The AI provider
            voice: Voice name (must be in AVAILABLE_VOICES)
        """
        if voice not in self.AVAILABLE_VOICES:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.AVAILABLE_VOICES}")
        
        self.voice_mapping[provider] = voice
    
    def generate_speech(self, text: str, voice: Optional[str] = None, 
                       model_name: Optional[str] = None) -> str:
        """
        Generate speech audio from text.
        
        Args:
            text: Text to convert to speech
            voice: Specific voice to use (overrides model-based selection)
            model_name: AI model name for voice selection (if voice not specified)
            
        Returns:
            Path to the generated audio file
            
        Raises:
            Exception: If TTS generation fails
        """
        if not text.strip():
            raise ValueError("Cannot generate speech for empty text")
        
        # Determine voice to use
        if voice:
            if voice not in self.AVAILABLE_VOICES:
                raise ValueError(f"Voice '{voice}' not available. Choose from: {self.AVAILABLE_VOICES}")
            selected_voice = voice
        elif model_name:
            selected_voice = self.get_voice_for_model(model_name)
        else:
            selected_voice = "alloy"  # Default voice
        
        try:
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice=selected_voice,
                input=text
            )
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", 
                dir=self.audio_dir, 
                delete=False
            ) as audio_file:
                response.stream_to_file(audio_file.name)
                return audio_file.name
                
        except Exception as e:
            raise Exception(f"TTS generation failed: {str(e)}")
    
    def generate_speech_for_conversation_turn(self, text: str, speaker_model: str) -> str:
        """
        Generate speech for a conversation turn with appropriate voice.
        
        Args:
            text: The conversation message text
            speaker_model: The AI model that generated this text
            
        Returns:
            Path to the generated audio file
        """
        return self.generate_speech(text, model_name=speaker_model)
    
    def get_voice_info(self) -> Dict[str, str]:
        """
        Get information about available voices and current mapping.
        
        Returns:
            Dictionary with voice information
        """
        return {
            "available_voices": self.AVAILABLE_VOICES,
            "current_mapping": {
                provider.value: voice 
                for provider, voice in self.voice_mapping.items()
            }
        }
    
    def cleanup_audio_files(self, older_than_hours: int = 24) -> int:
        """
        Clean up old audio files.
        
        Args:
            older_than_hours: Remove files older than this many hours
            
        Returns:
            Number of files cleaned up
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        cleaned_count = 0
        
        for audio_file in self.audio_dir.glob("*.mp3"):
            if audio_file.stat().st_mtime < cutoff_time:
                try:
                    audio_file.unlink()
                    cleaned_count += 1
                except OSError:
                    pass  # Ignore files that can't be deleted
        
        return cleaned_count


# Global TTS service instance
tts_service = None


def get_tts_service() -> TTSService:
    """Get the global TTS service instance."""
    global tts_service
    if tts_service is None:
        tts_service = TTSService()
    return tts_service


def generate_speech_for_text(text: str, model_name: Optional[str] = None, 
                           voice: Optional[str] = None) -> str:
    """
    Convenience function to generate speech for text.
    
    Args:
        text: Text to convert to speech
        model_name: AI model name for voice selection
        voice: Specific voice to use
        
    Returns:
        Path to generated audio file
    """
    service = get_tts_service()
    return service.generate_speech(text, voice=voice, model_name=model_name)


def cleanup_old_audio_files(hours: int = 24) -> int:
    """
    Convenience function to cleanup old audio files.
    
    Args:
        hours: Remove files older than this many hours
        
    Returns:
        Number of files cleaned up
    """
    service = get_tts_service()
    return service.cleanup_audio_files(hours)