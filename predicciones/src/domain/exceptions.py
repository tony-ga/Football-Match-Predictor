"""
Custom exceptions for ESPN integration and match prediction.

This module defines specific exception types for better error handling
and user-friendly error messages.
"""
from __future__ import annotations


class EspnApiError(Exception):
    """
    Raised when ESPN API request fails.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code if available
        url: The API URL that failed
        retry_after: Suggested retry delay in seconds if available
    """
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        url: str | None = None,
        retry_after: int | None = None
    ):
        self.message = message
        self.status_code = status_code
        self.url = url
        self.retry_after = retry_after
        super().__init__(self.message)
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")
        if self.url:
            parts.append(f"URL: {self.url}")
        return " ".join(parts)


class EspnParseError(Exception):
    """
    Raised when ESPN response cannot be parsed.
    
    Attributes:
        message: Human-readable error message
        field: The field that failed to parse
        raw_value: The raw value that caused the error
    """
    def __init__(
        self,
        message: str,
        field: str | None = None,
        raw_value: any = None
    ):
        self.message = message
        self.field = field
        self.raw_value = raw_value
        super().__init__(self.message)
    
    def __str__(self) -> str:
        msg = self.message
        if self.field:
            msg += f" (field: {self.field})"
        return msg


class MatchSelectionError(Exception):
    """
    Raised when match selection fails.
    
    Attributes:
        message: Human-readable error message
        available_matches: List of available matches if applicable
    """
    def __init__(
        self,
        message: str,
        available_matches: list | None = None
    ):
        self.message = message
        self.available_matches = available_matches or []
        super().__init__(self.message)


class MatchInputBuildError(Exception):
    """
    Raised when building match input fails.
    
    Attributes:
        message: Human-readable error message
        source: The source of data that failed (espn, json, teams)
        details: Additional error details
    """
    def __init__(
        self,
        message: str,
        source: str | None = None,
        details: dict | None = None
    ):
        self.message = message
        self.source = source
        self.details = details or {}
        super().__init__(self.message)


class TeamNotFoundError(Exception):
    """
    Raised when a team cannot be found or matched.
    
    Attributes:
        message: Human-readable error message
        searched_name: The team name that was searched
        alternatives: List of alternative team names
    """
    def __init__(
        self,
        message: str,
        searched_name: str | None = None,
        alternatives: list | None = None
    ):
        self.message = message
        self.searched_name = searched_name
        self.alternatives = alternatives or []
        super().__init__(self.message)
