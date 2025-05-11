"""
exceptions.py - Custom exceptions for the wdbx package.
"""


class WDBXError(Exception):
    """Base exception for WDBX wrapper errors."""


class RateLimitError(WDBXError):
    """Raised when a rate limit is exceeded."""


class AuthenticationError(WDBXError):
    """Raised for authentication failures."""


class AuthorizationError(WDBXError):
    """Raised for authorization failures."""
