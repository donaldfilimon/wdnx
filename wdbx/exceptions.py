"""
exceptions.py - Custom exceptions for the wdbx package.
"""

class WDBXError(Exception):
    """Base exception for WDBX wrapper errors."""
    pass

class RateLimitError(WDBXError):
    """Raised when a rate limit is exceeded."""
    pass

class AuthenticationError(WDBXError):
    """Raised for authentication failures."""
    pass

class AuthorizationError(WDBXError):
    """Raised for authorization failures."""
    pass 