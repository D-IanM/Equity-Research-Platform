"""Custom exception hierarchy for the project."""


class EquityResearchError(Exception):
    """Base exception for all project-specific failures."""


class DataSourceError(EquityResearchError):
    """Raised when an external data source cannot be queried or parsed."""


class ValidationError(EquityResearchError):
    """Raised when user input or downloaded data fails validation."""


class InsufficientDataError(EquityResearchError):
    """Raised when a calculation cannot be completed with available data."""
