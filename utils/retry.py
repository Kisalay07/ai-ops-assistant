from __future__ import annotations

from typing import Callable, TypeVar, Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")

class RetryableError(RuntimeError):
    """Raised for transient errors where retrying is appropriate."""

def with_retry(
    attempts: int = 3,
    min_seconds: float = 0.5,
    max_seconds: float = 6.0,
    retry_exceptions: tuple[type[BaseException], ...] = (RetryableError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator factory for exponential backoff + jitter retries."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=min_seconds, max=max_seconds),
        retry=retry_if_exception_type(retry_exceptions),
    )
