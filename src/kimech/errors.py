"""Public exceptions raised by Kimech."""


class KimechError(Exception):
    """Base class for Kimech-specific errors."""


class InvalidModelError(KimechError):
    """Raised when a mechanism or solve problem is structurally invalid."""


class KinematicSolveError(KimechError):
    """Raised when a kinematic configuration cannot be solved reliably."""
