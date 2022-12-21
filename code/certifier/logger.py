from .constants import DEBUG, VERBOSE


def print_if(msg: str, condition: bool):
    """Print `msg` if `condition` is True."""
    if condition:
        print(msg)


def dprint(msg: str):
    """Print `msg` if `DEBUG` is True."""
    print_if(msg, DEBUG and VERBOSE)
