"""
Structured logging for media pipeline operations.
Provides consistent logging across all services.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json


class StructuredLogger:
    """Structured logging with JSON format."""

    def __init__(self, service_name: str = "media-pipeline"):
        self.service_name = service_name
        self.logs = []

    def log(self, level: str, message: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Log a message with context."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "service": self.service_name,
            "message": message,
            "context": context or {},
            **kwargs
        }

        self.logs.append(entry)
        # Print for visibility
        print(json.dumps(entry))

        return entry

    def info(self, message: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Log info message."""
        return self.log("info", message, context, **kwargs)

    def warning(self, message: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Log warning message."""
        return self.log("warning", message, context, **kwargs)

    def error(self, message: str, context: Dict[str, Any] = None, error: str = None, **kwargs) -> Dict[str, Any]:
        """Log error message."""
        return self.log("error", message, context, error=error, **kwargs)

    def debug(self, message: str, context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Log debug message."""
        return self.log("debug", message, context, **kwargs)

    def get_logs(self, level: Optional[str] = None) -> list:
        """Retrieve logs, optionally filtered by level."""
        if level:
            return [log for log in self.logs if log["level"] == level.upper()]
        return self.logs

    def clear_logs(self) -> None:
        """Clear all logs."""
        self.logs.clear()

    def export_logs(self, format: str = "json") -> str:
        """Export logs in specified format."""
        if format == "json":
            return json.dumps(self.logs, indent=2)
        elif format == "csv":
            # Simple CSV format
            lines = ["timestamp,level,service,message"]
            for log in self.logs:
                lines.append(f"{log['timestamp']},{log['level']},{log['service']},{log['message']}")
            return "\n".join(lines)
        else:
            return str(self.logs)


class JobLogger:
    """Logger for tracking processing job lifecycle."""

    def __init__(self, job_id: str, service_name: str = "media-pipeline"):
        self.job_id = job_id
        self.service_name = service_name
        self.logger = StructuredLogger(service_name)
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """Log job start."""
        self.start_time = datetime.now(timezone.utc)
        self.logger.info(f"Job {self.job_id} started", {
            "job_id": self.job_id,
            "start_time": self.start_time.isoformat()
        })

    def progress(self, operation: str, progress: float, details: Dict[str, Any] = None) -> None:
        """Log job progress."""
        self.logger.info(f"Job {self.job_id} progress", {
            "job_id": self.job_id,
            "operation": operation,
            "progress": progress,
            "details": details or {}
        })

    def complete(self, result: Dict[str, Any] = None) -> None:
        """Log job completion."""
        self.end_time = datetime.now(timezone.utc)
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        self.logger.info(f"Job {self.job_id} completed", {
            "job_id": self.job_id,
            "duration_seconds": duration,
            "result": result or {}
        })

    def error(self, operation: str, error_message: str, details: Dict[str, Any] = None) -> None:
        """Log job error."""
        self.logger.error(f"Job {self.job_id} error", {
            "job_id": self.job_id,
            "operation": operation,
            "details": details or {}
        }, error=error_message)

    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return None

    def export_logs(self) -> list:
        """Export job logs."""
        return self.logger.get_logs()
