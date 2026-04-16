import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


def safe_call(fn: Callable[..., Any], *args, default: Any = None, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.exception("safe_call caught exception: %s", e)
        return default
