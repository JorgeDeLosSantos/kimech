"""Kimech: planar mechanism kinematics for Python."""

from .diagnostics import SolveDiagnosticSummary, SolveDiagnostics
from .errors import (
    InvalidModelError,
    KimechError,
    KinematicSolveError,
    SolveFailureContext,
)
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism, Point
from .solver import solve
from .solution import Configuration, KinematicSolution
from .topology import MechanismTopology
from .validation import ValidationReport

__all__ = [
    "Configuration",
    "Ground",
    "InvalidModelError",
    "KimechError",
    "KinematicSolveError",
    "KinematicSolution",
    "Link",
    "Mechanism",
    "MechanismTopology",
    "Point",
    "PrismaticJoint",
    "RevoluteJoint",
    "SolveDiagnosticSummary",
    "SolveDiagnostics",
    "SolveFailureContext",
    "solve",
    "ValidationReport",
]

__version__ = "0.4.0"
