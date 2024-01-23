from structlog import make_filtering_bound_logger, wrap_logger
from structlog.typing import FilteringBoundLogger
from structlog._log_levels import NAME_TO_LEVEL
from enum import Enum


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
