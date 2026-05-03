"""
utils/helpers.py
----------------
General-purpose helper functions used across all FinAgent agents.
Provides safe data access, currency/percentage formatting, and
human-readable number conversion.
"""

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Safe data access
# ---------------------------------------------------------------------------

def safe_get(data: dict, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a dictionary.

    Parameters
    ----------
    data    : The source dictionary.
    key     : The key to look up.
    default : Value returned when key is missing or value is None / NaN.

    Returns
    -------
    The value if present and not None/NaN, otherwise `default`.
    """
    try:
        value = data.get(key, default)
        # Treat pandas NA / float NaN as missing
        if value is None:
            return default
        # Check for float NaN without importing pandas here
        if isinstance(value, float) and (value != value):   # NaN != NaN is True
            return default
        return value
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_currency(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a numeric value as a USD currency string.

    Examples
    --------
    >>> format_currency(1_500_000_000)
    '$1.50B'
    >>> format_currency(250_000)
    '$250,000.00'
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if abs(value) >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value:,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """
    Format a decimal fraction (e.g. 0.25) as a percentage string ("25.00%").

    If the value is already expressed as a percentage (> 1 by convention here
    we treat any value as a fraction), callers should divide by 100 first.
    """
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_large_number(value: Optional[float]) -> str:
    """
    Convert a large number to a short human-readable form.

    Examples
    --------
    >>> format_large_number(2_300_000_000_000)
    '2.30T'
    """
    if value is None:
        return "N/A"
    try:
        value = float(value)
        if abs(value) >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f}T"
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.2f}K"
        return f"{value:.2f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_ticker(ticker: str) -> bool:
    """
    Basic validation that a ticker string looks plausible.

    Rules
    -----
    - 1 to 10 alphanumeric characters (allowing dots and hyphens for
      tickers like BRK.B or BF-B).
    - Not empty after stripping whitespace.
    """
    import re
    if not ticker or not ticker.strip():
        return False
    pattern = r"^[A-Za-z0-9.\-]{1,10}$"
    return bool(re.match(pattern, ticker.strip()))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp `value` to the range [min_val, max_val]."""
    return max(min_val, min(max_val, value))
