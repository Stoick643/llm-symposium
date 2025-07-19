"""
Audio Management System

Handles audio file operations, playback, and management for the TTS system.
"""

import os
import subprocess
import platform
from typing import Optional, List
from pathlib import Path


class AudioManager:
    """Manages audio file operations and playback."""
    
    def __init__(self, audio_directory: str = "audio_cache"):
        """
        Initialize the audio manager.
        
        Args:
            audio_directory: Directory to store audio files
        """
        self.audio_dir = Path(audio_directory)
        self.audio_dir.mkdir(exist_ok=True)
        
        # Detect system audio capabilities
        self.audio_player = self._detect_audio_player()
    
    def _detect_audio_player(self) -> Optional[str]:
        """
        Detect available audio player on the system.
        
        Returns:
            Command for audio playback or None if not available
        """
        system = platform.system().lower()
        
        # Try different audio players based on system
        players_to_try = []
        
        if system == "linux":
            players_to_try = ["paplay", "aplay", "mpg123", "ffplay", "vlc"]
        elif system == "darwin":  # macOS
            players_to_try = ["afplay", "mpg123", "ffplay"]
        elif system == "windows":
            players_to_try = ["powershell", "ffplay"]
        
        for player in players_to_try:
            if self._command_exists(player):
                return player
        
        return None
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists on the system."""
        try:
            subprocess.run([command, "--help"], 
                         capture_output=True, 
                         check=False)
            return True
        except FileNotFoundError:
            return False
    
    def play_audio_file(self, audio_file_path: str, background: bool = True) -> bool:
        """
        Play an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            background: Whether to play in background (non-blocking)
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if not os.path.exists(audio_file_path):
            print(f"Audio file not found: {audio_file_path}")
            return False
        
        if not self.audio_player:
            print("No audio player available on this system")
            return False
        
        try:
            if self.audio_player == "afplay":  # macOS
                cmd = ["afplay", audio_file_path]
            elif self.audio_player in ["paplay", "aplay"]:  # Linux ALSA/PulseAudio
                cmd = [self.audio_player, audio_file_path]
            elif self.audio_player == "mpg123":
                cmd = ["mpg123", "-q", audio_file_path]  # -q for quiet
            elif self.audio_player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", audio_file_path]
            elif self.audio_player == "powershell":  # Windows
                # Use PowerShell's media player
                cmd = ["powershell", "-c", 
                      f"(New-Object Media.SoundPlayer '{audio_file_path}').PlaySync()"]
            elif self.audio_player == "vlc":
                cmd = ["vlc", "--intf", "dummy", "--play-and-exit", audio_file_path]
            else:
                print(f"Unknown audio player: {self.audio_player}")
                return False
            
            if background:
                # Start in background (non-blocking)
                subprocess.Popen(cmd, 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            else:
                # Wait for completion (blocking)
                subprocess.run(cmd, 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            
            return True
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def get_audio_info(self, audio_file_path: str) -> dict:
        """
        Get information about an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(audio_file_path):
            return {"exists": False}
        
        file_stat = os.stat(audio_file_path)
        
        return {
            "exists": True,
            "path": audio_file_path,
            "size_bytes": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "created": file_stat.st_ctime,
            "modified": file_stat.st_mtime
        }
    
    def list_audio_files(self) -> List[dict]:
        """
        List all audio files in the audio directory.
        
        Returns:
            List of dictionaries with file information
        """
        audio_files = []
        
        for audio_file in self.audio_dir.glob("*.mp3"):
            info = self.get_audio_info(str(audio_file))
            audio_files.append(info)
        
        # Sort by creation time (newest first)
        audio_files.sort(key=lambda x: x.get("created", 0), reverse=True)
        
        return audio_files
    
    def cleanup_audio_files(self, older_than_hours: int = 24) -> int:
        """
        Remove old audio files.
        
        Args:
            older_than_hours: Remove files older than this many hours
            
        Returns:
            Number of files removed
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        removed_count = 0
        
        for audio_file in self.audio_dir.glob("*.mp3"):
            try:
                if audio_file.stat().st_mtime < cutoff_time:
                    audio_file.unlink()
                    removed_count += 1
            except OSError:
                pass  # Ignore files that can't be deleted
        
        return removed_count
    
    def get_total_audio_size(self) -> dict:
        """
        Get total size of all audio files.
        
        Returns:
            Dictionary with size information
        """
        total_bytes = 0
        file_count = 0
        
        for audio_file in self.audio_dir.glob("*.mp3"):
            try:
                total_bytes += audio_file.stat().st_size
                file_count += 1
            except OSError:
                pass
        
        return {
            "total_files": file_count,
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 2)
        }
    
    def is_audio_available(self) -> bool:
        """
        Check if audio playback is available on this system.
        
        Returns:
            True if audio playback is supported
        """
        return self.audio_player is not None
    
    def get_system_info(self) -> dict:
        """
        Get system audio information.
        
        Returns:
            Dictionary with system audio capabilities
        """
        return {
            "system": platform.system(),
            "audio_player": self.audio_player,
            "audio_available": self.is_audio_available(),
            "audio_directory": str(self.audio_dir)
        }


# Global audio manager instance
audio_manager = None


def get_audio_manager() -> AudioManager:
    """Get the global audio manager instance."""
    global audio_manager
    if audio_manager is None:
        audio_manager = AudioManager()
    return audio_manager


def play_audio(audio_file_path: str, background: bool = True) -> bool:
    """
    Convenience function to play an audio file.
    
    Args:
        audio_file_path: Path to the audio file
        background: Whether to play in background
        
    Returns:
        True if playback started successfully
    """
    manager = get_audio_manager()
    return manager.play_audio_file(audio_file_path, background)


def cleanup_old_audio(hours: int = 24) -> int:
    """
    Convenience function to cleanup old audio files.
    
    Args:
        hours: Remove files older than this many hours
        
    Returns:
        Number of files cleaned up
    """
    manager = get_audio_manager()
    return manager.cleanup_audio_files(hours)