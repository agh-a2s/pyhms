from enum import Enum

from structlog import make_filtering_bound_logger, wrap_logger
from structlog._log_levels import NAME_TO_LEVEL
from structlog.typing import FilteringBoundLogger


class LoggingLevel(str, Enum):
    CRITICAL = "critical"
    EXCEPTION = "exception"
    ERROR = "error"
    WARN = "warn"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    NOTSET = "notset"


DEFAULT_LOGGING_LEVEL = LoggingLevel.WARNING


def get_logger(level: LoggingLevel = DEFAULT_LOGGING_LEVEL) -> FilteringBoundLogger:
    wrapper_class = make_filtering_bound_logger(NAME_TO_LEVEL[level])
    return wrap_logger(
        None,
        wrapper_class=wrapper_class,
    )


def parse_log_level(log_level: str | LoggingLevel | None) -> LoggingLevel:
    if log_level is None:
        return LoggingLevel.WARNING
    if isinstance(log_level, LoggingLevel):
        return log_level
    levels = [level.value for level in LoggingLevel]
    if log_level.lower() not in levels:
        raise ValueError(f"Invalid log level: {log_level}, available levels: {levels}")
    return LoggingLevel[log_level.upper()]
