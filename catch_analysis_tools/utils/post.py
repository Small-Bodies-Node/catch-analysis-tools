"""Functions for working with POST data."""

from typing import Any


def get_float(
    body: dict[str, Any],
    key: str,
    default: float | None,
    nullable: bool = True,
    exception_class: Exception = ValueError,
) -> float | None:
    """Get a floating point value.


    Parameters
    ----------
    body : dict
        The POST data.

    key : string
        The key to convert to a float.

    default : float or None
        A default value.

    nullable : bool, optional
        ``True`` if the value may be ``None``.

    exception_class : Exception, optional
        Raise an exception from this class if the value is invalid.


    Returns
    -------
    value : float or None

    """

    value = body.get(key, default)

    if value is None and nullable:
        return None
    elif value is None and not nullable:
        raise exception_class(f"{key} may not be null")

    try:
        return float(value)
    except (TypeError, ValueError):
        exception_class(f"{key} must be a valid number")


def get_bool(
    body: dict[str, Any],
    key: str,
    default: bool | None,
    nullable: bool = True,
    exception_class: Exception = ValueError,
) -> bool | None:
    """Get a boolean value.


    Parameters
    ----------
    body : dict
        The POST data.

    key : string
        The key to convert to a bool.

    default : float or None
        A default value.

    nullable : bool, optional
        ``True`` if the value may be ``None``.

    exception_class : Exception, optional
        Raise an exception from this class if the value is invalid.


    Returns
    -------
    value : bool or None

    """

    value = body.get(key, default)

    if value is None and nullable:
        return None
    elif value is None and not nullable:
        raise exception_class(f"{key} may not be null")

    try:
        if isinstance(value, str):
            if value.lower() in {"true", "1", "yes", "y"}:
                return True
            if value.lower() in {"false", "0", "no", "n"}:
                return False
            if value.isdigit():
                return bool(value)

        return bool(value)

    except (TypeError, ValueError):
        exception_class(
            f'{key} must be a valid boolean, e.g., "true", "false", 0, 1, "yes", "no"'
        )
