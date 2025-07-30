"""
Professional logging system for Project Hyperion
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
import io
import contextlib

from config.settings import settings

class TerminalCapture:
    """Capture terminal output and redirect to log files"""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_capture = None
        self.stderr_capture = None
        self.log_file = None
        
    def start_capture(self):
        """Start capturing terminal output"""
        try:
            # Open log file for writing
            self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)
            
            # Create custom stdout/stderr that writes to both terminal and file
            class TeeOutput:
                def __init__(self, original_stream, log_file):
                    self.original_stream = original_stream
                    self.log_file = log_file
                    self.buffer = ""
                
                def write(self, text):
                    # Write to original stream immediately
                    self.original_stream.write(text)
                    
                    # Add to buffer and flush to file
                    self.buffer += text
                    if '\n' in text or len(self.buffer) > 1024:
                        self.log_file.write(self.buffer)
                        self.log_file.flush()
                        self.buffer = ""
                
                def flush(self):
                    self.original_stream.flush()
                    if self.buffer:
                        self.log_file.write(self.buffer)
                        self.buffer = ""
                    self.log_file.flush()
            
            # Replace stdout and stderr
            self.stdout_capture = TeeOutput(sys.stdout, self.log_file)
            self.stderr_capture = TeeOutput(sys.stderr, self.log_file)
            sys.stdout = self.stdout_capture
            sys.stderr = self.stderr_capture
            
        except Exception as e:
            print(f"WARNING: Could not start terminal capture: {e}")
    
    def stop_capture(self):
        """Stop capturing terminal output"""
        try:
            # Flush any remaining buffer
            if self.stdout_capture:
                self.stdout_capture.flush()
            if self.stderr_capture:
                self.stderr_capture.flush()
            
            # Restore original stdout and stderr
            if self.stdout_capture:
                sys.stdout = self.original_stdout
            if self.stderr_capture:
                sys.stderr = self.original_stderr
            
            # Close log file
            if self.log_file:
                self.log_file.close()
                
        except Exception as e:
            print(f"WARNING: Could not stop terminal capture: {e}")

class SessionLogger:
    """
    Session-based logging system that organizes all logs into session folders
    """
    
    def __init__(self):
        self.session_id = None
        self.session_dir = None
        self.session_start_time = None
        self.loggers = {}
        self.session_info = {}
        self.terminal_capture = None
    
    def start_session(self, session_name: str = None) -> str:
        """
        Start a new logging session
        
        Args:
            session_name: Optional custom session name
            
        Returns:
            Session ID
        """
        # Generate session ID
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start_time = datetime.now()
        
        # Create session name
        if session_name:
            session_name = session_name.replace(' ', '_').replace('/', '_')
            session_dir_name = f"{session_name}_{self.session_id}_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        else:
            session_dir_name = f"session_{self.session_id}_{self.session_start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create session directory
        self.session_dir = settings.LOGS_DIR / session_dir_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session info
        self.session_info = {
            'session_id': self.session_id,
            'session_name': session_name or 'default',
            'start_time': self.session_start_time,
            'session_dir': str(self.session_dir),
            'components': []
        }
        
        # Create session info file
        self._save_session_info()
        
        # Start terminal capture
        terminal_log_path = self.session_dir / 'terminal_output.log'
        self.terminal_capture = TerminalCapture(terminal_log_path)
        self.terminal_capture.start_capture()
        
        print(f"📁 Logging session started: {session_dir_name}")
        print(f"📂 Session directory: {self.session_dir}")
        print(f"📝 Terminal output will be saved to: {terminal_log_path}")
        
        return self.session_id
    
    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        """
        Get a logger for the current session
        
        Args:
            name: Logger name
            level: Logging level
            
        Returns:
            Logger instance
        """
        if not self.session_id:
            raise RuntimeError("No active session. Call start_session() first.")
        
        # Check if logger already exists
        if name in self.loggers:
            return self.loggers[name]
        
        # Create new logger
        logger = self._create_session_logger(name, level)
        self.loggers[name] = logger
        
        # Add to session info
        self.session_info['components'].append({
            'name': name,
            'level': level,
            'created_at': datetime.now().isoformat()
        })
        self._save_session_info()
        
        return logger
    
    def _create_session_logger(self, name: str, level: str) -> logging.Logger:
        """Create a logger for the current session"""
        logger = logging.getLogger(f"hyperion.{name}")
        logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Main log file handler
        try:
            main_log_file = self.session_dir / f'{name}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                main_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"WARNING: Could not create main log file handler: {e}")
        
        # Error log file handler
        try:
            error_log_file = self.session_dir / f'{name}_errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            logger.addHandler(error_handler)
        except Exception as e:
            print(f"WARNING: Could not create error log file handler: {e}")
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Prevent propagation
        logger.propagate = False
        
        return logger
    
    def _save_session_info(self):
        """Save session information to file"""
        try:
            import json
            info_file = self.session_dir / 'session_info.json'
            with open(info_file, 'w') as f:
                json.dump(self.session_info, f, indent=2, default=str)
        except Exception as e:
            print(f"WARNING: Could not save session info: {e}")
    
    def end_session(self):
        """End the current session"""
        if not self.session_id:
            return
        
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds()
        
        # Update session info
        self.session_info['end_time'] = end_time
        self.session_info['duration_seconds'] = duration
        self.session_info['total_loggers'] = len(self.loggers)
        self._save_session_info()
        
        # Log session summary
        summary_logger = self.get_logger('session_summary')
        summary_logger.info("="*80)
        summary_logger.info("SESSION SUMMARY")
        summary_logger.info("="*80)
        summary_logger.info(f"Session ID: {self.session_id}")
        summary_logger.info(f"Session Name: {self.session_info['session_name']}")
        summary_logger.info(f"Start Time: {self.session_start_time}")
        summary_logger.info(f"End Time: {end_time}")
        summary_logger.info(f"Duration: {duration:.2f} seconds")
        summary_logger.info(f"Total Loggers: {len(self.loggers)}")
        summary_logger.info(f"Session Directory: {self.session_dir}")
        summary_logger.info("="*80)
        
        # Stop terminal capture
        if self.terminal_capture:
            self.terminal_capture.stop_capture()
        
        # Close all handlers
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        
        print(f"📁 Logging session ended: {self.session_id}")
        print(f"📂 Session directory: {self.session_dir}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📝 Terminal output saved to: {self.session_dir}/terminal_output.log")
        
        # Reset session
        self.session_id = None
        self.session_dir = None
        self.session_start_time = None
        self.loggers = {}
        self.session_info = {}
        self.terminal_capture = None
    
    def get_session_directory(self) -> Optional[Path]:
        """Get the current session directory"""
        return self.session_dir
    
    def is_session_active(self) -> bool:
        """Check if a session is currently active"""
        return self.session_id is not None

# Global session logger instance
session_logger = SessionLogger()

def start_logging_session(session_name: str = None) -> str:
    """
    Start a new logging session
    
    Args:
        session_name: Optional custom session name
        
    Returns:
        Session ID
    """
    return session_logger.start_session(session_name)

def get_session_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger for the current session
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Logger instance
    """
    return session_logger.get_logger(name, level)

def end_logging_session():
    """End the current logging session"""
    session_logger.end_session()

def get_session_directory() -> Optional[Path]:
    """Get the current session directory"""
    return session_logger.get_session_directory()

def is_session_active() -> bool:
    """Check if a session is currently active"""
    return session_logger.is_session_active()

# Legacy functions for backward compatibility
def setup_logger(
    name: str = "hyperion",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    rotation_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup professional logging with rotation and multiple handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Log directory (defaults to settings)
        rotation_size: Max size for log rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # If session is active, use session logger
    if session_logger.is_session_active():
        return session_logger.get_logger(name, level)
    
    # Fallback to legacy logging
    # Use settings if not provided
    if log_dir is None:
        log_dir = settings.LOGS_DIR
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Main log file handler with rotation
    try:
        main_log_file = log_dir / f'{name}_{timestamp}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=rotation_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"WARNING: Could not create main log file handler: {e}")
    
    # Error log file handler (for critical errors only)
    try:
        error_log_file = log_dir / f'{name}_errors_{timestamp}.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=rotation_size // 2,  # 5MB
            backupCount=backup_count // 2,  # 2 backups
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    except Exception as e:
        print(f"WARNING: Could not create error log file handler: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = "hyperion") -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    # If session is active, use session logger
    if session_logger.is_session_active():
        return session_logger.get_logger(name)
    
    # Fallback to legacy logging
    logger = logging.getLogger(name)
    
    # If logger has no handlers, setup default logging
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger

def setup_training_logger(training_mode: str) -> logging.Logger:
    """
    Setup specialized logger for training operations
    
    Args:
        training_mode: Name of the training mode
        
    Returns:
        Training logger instance
    """
    logger_name = f"training.{training_mode}"
    return get_session_logger(logger_name) if session_logger.is_session_active() else setup_logger(f"hyperion.training.{training_mode}")

def setup_model_logger(model_name: str) -> logging.Logger:
    """
    Setup specialized logger for model operations
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model logger instance
    """
    logger_name = f"model.{model_name}"
    return get_session_logger(logger_name) if session_logger.is_session_active() else setup_logger(f"hyperion.model.{model_name}")

def setup_api_logger() -> logging.Logger:
    """
    Setup specialized logger for API operations
    
    Returns:
        API logger instance
    """
    return get_session_logger("api") if session_logger.is_session_active() else setup_logger("hyperion.api")

def log_system_info(logger: logging.Logger):
    """Log system information"""
    logger.info("="*80)
    logger.info("PROJECT HYPERION SYSTEM STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log directory: {settings.LOGS_DIR}")
    logger.info(f"Data directory: {settings.DATA_DIR}")
    logger.info(f"Models directory: {settings.MODELS_DIR}")
    if session_logger.is_session_active():
        logger.info(f"Session directory: {session_logger.get_session_directory()}")
    logger.info("="*80)

def log_training_start(logger: logging.Logger, mode: str, symbols: list):
    """Log training start information"""
    logger.info("🚀 TRAINING SESSION STARTED")
    logger.info(f"Mode: {mode}")
    logger.info(f"Symbols: {len(symbols)} pairs")
    logger.info(f"Symbol list: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

def log_training_complete(logger: logging.Logger, duration: float, models_trained: int, metrics: dict):
    """Log training completion information"""
    logger.info("✅ TRAINING SESSION COMPLETED")
    logger.info(f"Duration: {duration:.2f} seconds")
    logger.info(f"Models trained: {models_trained}")
    if metrics:
        avg_metric = sum(metrics.values()) / len(metrics)
        logger.info(f"Average performance: {avg_metric:.4f}")

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context"""
    logger.error(f"❌ ERROR in {context}: {error}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error details: {str(error)}")

# Setup default logger
default_logger = setup_logger("hyperion")
log_system_info(default_logger) 