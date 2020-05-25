"""Miscellaneous things.
"""


import sys
# We're using pytest to run tests. It's set up so that when tests are
# being run, pytest will set the attribute '_called_from_test' on the
# sys module. This means we can tell whether we're inside a testing
# session by checking whether this attribute exists. IS_TEST is a
# global boolean variable that denotes whether we're in a pytest run.
# Pytest documentation:
# https://docs.pytest.org/en/latest/example/simple.html#detect-if-running-from-within-a-pytest-run
IS_TEST = False
if hasattr(sys, '_called_from_test'):
    IS_TEST = True


def printif(condition, *args, **kwargs):
    """Wrapper function for print. Only prints if condition is met.

    condition will be treated as a boolean. If it is True, the rest of
    the parameters will be forwarded to print.
    Usually, condition will be "verbose >= some value".
    """
    if condition:
        print(*args, **kwargs)


# Below are global constants for verbosity levels.

DEBUGLVL = 4
"""Very verbose output, mainly for use when debugging/testing."""

USERLVL = 1
"""User-level verbosity: less verbose, intended for the end user."""
