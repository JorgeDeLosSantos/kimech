"""Kimech: planar mechanism kinematics for Python."""

from .diagnostics import SolveDiagnosticSummary, SolveDiagnostics
from .driver import KinematicDriver
from .errors import (
    InvalidModelError,
    KimechError,
    KinematicSolveError,
    SolveFailureContext,
)
from .joints import PrismaticJoint, RevoluteJoint
from .model import Ground, Link, Mechanism, Point
from .sensitivity import InputSensitivity, input_sensitivity
from .solver import solve
from .solution import Configuration, KinematicSolution
from .topology import MechanismTopology
from .validation import ValidationReport

__all__ = [
    "Configuration",
    "Ground",
    "InputSensitivity",
    "InvalidModelError",
    "KimechError",
    "KinematicDriver",
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
    "input_sensitivity",
    "solve",
    "ValidationReport",
]

__version__ = "0.5.0"
