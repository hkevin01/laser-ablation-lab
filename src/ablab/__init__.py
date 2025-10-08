"""
Laser Ablation Lab - Open models and mission trades for laser ablation 
and laser-induced disruption of small bodies.

This package provides comprehensive physics-based simulation tools for studying
laser ablation and laser-induced disruption of asteroids and comets for 
planetary defense applications.

Modules:
    constants: Physical constants and unit definitions
    units: Unit conversion and dimensional analysis
    geometry: Shape models and mesh generation for target bodies
    optics: Laser beam profiles, pointing, and thermal management
    physics: Core physics models (heat transfer, ablation, momentum coupling)
    dynamics: Orbital and rotational dynamics, engagement geometry
    mission: Mission analysis, sizing, and trade studies
    io: Configuration management and data persistence
    viz: Visualization and animation tools
"""

__version__ = "0.1.0"
__author__ = "Laser Ablation Lab Contributors"
__license__ = "Apache-2.0"

# Import core functionality
from . import constants
from . import units
from . import geometry
from . import optics
from . import physics
from . import dynamics
from . import mission
from . import io
from . import viz

__all__ = [
    "constants",
    "units", 
    "geometry",
    "optics",
    "physics",
    "dynamics", 
    "mission",
    "io",
    "viz"
]
