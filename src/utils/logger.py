"""
Structured logging for GPU optimization tools.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


# Global loggers registry
_loggers = {}


def setup_logger(
    name: str = "gpu_optimize",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rich_output: bool = True,
) -> logging.Logger:
    """
    Set up a structured logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        rich_output: Use rich formatting for console
        
    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    if rich_output:
        console_handler = RichHandler(
            console=Console(),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "gpu_optimize") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class MetricsLogger:
    """
    Logger for performance metrics.
    
    Example:
        >>> metrics = MetricsLogger("training")
        >>> metrics.log("epoch", 1)
        >>> metrics.log("loss", 0.5)
        >>> metrics.log("gpu_util", 92.5)
        >>> metrics.summary()
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            name: Name for this metrics log
            log_file: Optional file to write metrics
        """
        self.name = name
        self.log_file = log_file
        self._metrics = {}
        self._start_time = datetime.now()
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if key not in self._metrics:
            self._metrics[key] = []
        
        entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
        
        if step is not None:
            entry["step"] = step
        
        self._metrics[key].append(entry)
        
        # Write to file if configured
        if self.log_file:
            with open(self.log_file, "a") as f:
                step_str = f", step={step}" if step is not None else ""
                f.write(f"{entry['timestamp']}, {key}={value:.6f}{step_str}\n")
    
    def get(self, key: str) -> list:
        """Get all values for a metric."""
        return [e["value"] for e in self._metrics.get(key, [])]
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get latest value for a metric."""
        values = self.get(key)
        return values[-1] if values else None
    
    def get_mean(self, key: str) -> Optional[float]:
        """Get mean value for a metric."""
        values = self.get(key)
        return sum(values) / len(values) if values else None
    
    def summary(self) -> str:
        """Generate metrics summary."""
        duration = (datetime.now() - self._start_time).total_seconds()
        
        lines = [
            f"Metrics Summary: {self.name}",
            f"Duration: {duration:.2f}s",
            "",
        ]
        
        for key, entries in self._metrics.items():
            values = [e["value"] for e in entries]
            lines.append(
                f"  {key}:"
                f"\n    Count: {len(values)}"
                f"\n    Mean: {sum(values)/len(values):.4f}"
                f"\n    Min: {min(values):.4f}"
                f"\n    Max: {max(values):.4f}"
                f"\n    Latest: {values[-1]:.4f}"
            )
        
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        print(self.summary())
