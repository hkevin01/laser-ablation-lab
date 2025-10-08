"""
Comprehensive units and dimensional analysis system for laser ablation simulations.

This module provides automatic unit conversion, dimensional analysis validation,
and uncertainty propagation for all physical quantities used in laser ablation
simulations. It ensures consistency and helps prevent unit-related errors.

Features:
    - Automatic unit conversion between SI, CGS, and Imperial systems
    - Dimensional analysis validation for physics equations
    - Uncertainty propagation using error analysis
    - Unit-aware arithmetic operations
    - Clear error messages for unit mismatches

References:
    - NIST Guide to SI Units
    - ISO 80000 series (Quantities and units)
    - Taylor & Kuyatt (1994) Guidelines for Evaluating and Expressing Uncertainty
"""

import numpy as np
from typing import Union, Dict, Any, Optional, Tuple
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class Dimension(Enum):
    """Enumeration of fundamental physical dimensions."""
    LENGTH = "L"
    MASS = "M" 
    TIME = "T"
    TEMPERATURE = "Θ"
    CURRENT = "I"
    LUMINOUS_INTENSITY = "J"
    AMOUNT = "N"
    ANGLE = "A"
    DIMENSIONLESS = "1"


@dataclass
class UnitDimensions:
    """Container for dimensional analysis using fundamental dimensions."""
    length: float = 0.0      # L
    mass: float = 0.0        # M
    time: float = 0.0        # T
    temperature: float = 0.0 # Θ (theta)
    current: float = 0.0     # I
    luminous: float = 0.0    # J
    amount: float = 0.0      # N
    angle: float = 0.0       # A
    
    def __eq__(self, other) -> bool:
        """Check dimensional equality with tolerance for floating point errors."""
        if not isinstance(other, UnitDimensions):
            return False
        
        tolerance = 1e-10
        return (abs(self.length - other.length) < tolerance and
                abs(self.mass - other.mass) < tolerance and
                abs(self.time - other.time) < tolerance and
                abs(self.temperature - other.temperature) < tolerance and
                abs(self.current - other.current) < tolerance and
                abs(self.luminous - other.luminous) < tolerance and
                abs(self.amount - other.amount) < tolerance and
                abs(self.angle - other.angle) < tolerance)
    
    def __mul__(self, other) -> 'UnitDimensions':
        """Multiply dimensions (for unit multiplication)."""
        return UnitDimensions(
            length=self.length + other.length,
            mass=self.mass + other.mass,
            time=self.time + other.time,
            temperature=self.temperature + other.temperature,
            current=self.current + other.current,
            luminous=self.luminous + other.luminous,
            amount=self.amount + other.amount,
            angle=self.angle + other.angle
        )
    
    def __truediv__(self, other) -> 'UnitDimensions':
        """Divide dimensions (for unit division)."""
        return UnitDimensions(
            length=self.length - other.length,
            mass=self.mass - other.mass,
            time=self.time - other.time,
            temperature=self.temperature - other.temperature,
            current=self.current - other.current,
            luminous=self.luminous - other.luminous,
            amount=self.amount - other.amount,
            angle=self.angle - other.angle
        )
    
    def __pow__(self, exponent: float) -> 'UnitDimensions':
        """Raise dimensions to a power."""
        return UnitDimensions(
            length=self.length * exponent,
            mass=self.mass * exponent,
            time=self.time * exponent,
            temperature=self.temperature * exponent,
            current=self.current * exponent,
            luminous=self.luminous * exponent,
            amount=self.amount * exponent,
            angle=self.angle * exponent
        )
    
    def __str__(self) -> str:
        """String representation of dimensions."""
        parts = []
        dim_names = ['L', 'M', 'T', 'Θ', 'I', 'J', 'N', 'A']
        values = [self.length, self.mass, self.time, self.temperature,
                 self.current, self.luminous, self.amount, self.angle]
        
        for name, value in zip(dim_names, values):
            if abs(value) > 1e-10:
                if abs(value - 1.0) < 1e-10:
                    parts.append(name)
                elif abs(value + 1.0) < 1e-10:
                    parts.append(f"{name}⁻¹")
                else:
                    parts.append(f"{name}^{value:.2g}")
        
        return " ".join(parts) if parts else "1"


@dataclass
class Quantity:
    """
    Physical quantity with value, unit, uncertainty, and dimensional analysis.
    
    This class represents a physical quantity with automatic unit conversion,
    dimensional analysis, and uncertainty propagation.
    """
    value: float
    unit: str
    uncertainty: Optional[float] = None
    dimensions: Optional[UnitDimensions] = None
    
    def __post_init__(self):
        """Initialize dimensions if not provided."""
        if self.dimensions is None:
            self.dimensions = get_unit_dimensions(self.unit)
        
        # Validate uncertainty
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative")
    
    def to(self, target_unit: str) -> 'Quantity':
        """
        Convert to a different unit with dimensional validation.
        
        Args:
            target_unit: Target unit string
            
        Returns:
            New Quantity object in target units
            
        Raises:
            ValueError: If units are dimensionally incompatible
        """
        target_dimensions = get_unit_dimensions(target_unit)
        
        if not self.dimensions == target_dimensions:
            raise ValueError(
                f"Cannot convert from {self.unit} [{self.dimensions}] "
                f"to {target_unit} [{target_dimensions}] - incompatible dimensions"
            )
        
        conversion_factor = convert_units(1.0, self.unit, target_unit)
        new_value = self.value * conversion_factor
        new_uncertainty = self.uncertainty * conversion_factor if self.uncertainty else None
        
        return Quantity(new_value, target_unit, new_uncertainty, target_dimensions)
    
    def __add__(self, other: 'Quantity') -> 'Quantity':
        """Add two quantities with dimensional validation."""
        if not self.dimensions == other.dimensions:
            raise ValueError(
                f"Cannot add quantities with incompatible dimensions: "
                f"{self.dimensions} + {other.dimensions}"
            )
        
        # Convert other to self's units
        other_converted = other.to(self.unit)
        
        new_value = self.value + other_converted.value
        new_uncertainty = None
        if self.uncertainty and other_converted.uncertainty:
            # Uncertainty propagation for addition
            new_uncertainty = np.sqrt(self.uncertainty**2 + other_converted.uncertainty**2)
        elif self.uncertainty:
            new_uncertainty = self.uncertainty
        elif other_converted.uncertainty:
            new_uncertainty = other_converted.uncertainty
        
        return Quantity(new_value, self.unit, new_uncertainty, self.dimensions)
    
    def __sub__(self, other: 'Quantity') -> 'Quantity':
        """Subtract two quantities with dimensional validation."""
        return self + (other * -1)
    
    def __mul__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        """Multiply quantity by another quantity or scalar."""
        if isinstance(other, (int, float)):
            new_value = self.value * other
            new_uncertainty = self.uncertainty * abs(other) if self.uncertainty else None
            return Quantity(new_value, self.unit, new_uncertainty, self.dimensions)
        
        elif isinstance(other, Quantity):
            new_value = self.value * other.value
            new_dimensions = self.dimensions * other.dimensions
            new_unit = f"({self.unit})⋅({other.unit})"
            
            # Uncertainty propagation for multiplication
            new_uncertainty = None
            if self.uncertainty and other.uncertainty:
                rel_unc_self = self.uncertainty / abs(self.value) if self.value != 0 else 0
                rel_unc_other = other.uncertainty / abs(other.value) if other.value != 0 else 0
                rel_unc_result = np.sqrt(rel_unc_self**2 + rel_unc_other**2)
                new_uncertainty = abs(new_value) * rel_unc_result
            
            return Quantity(new_value, new_unit, new_uncertainty, new_dimensions)
        
        else:
            raise TypeError(f"Cannot multiply Quantity by {type(other)}")
    
    def __rmul__(self, other: Union[float, int]) -> 'Quantity':
        """Right multiplication for scalars."""
        return self * other
    
    def __truediv__(self, other: Union['Quantity', float, int]) -> 'Quantity':
        """Divide quantity by another quantity or scalar."""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            new_value = self.value / other
            new_uncertainty = self.uncertainty / abs(other) if self.uncertainty else None
            return Quantity(new_value, self.unit, new_uncertainty, self.dimensions)
        
        elif isinstance(other, Quantity):
            if other.value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            
            new_value = self.value / other.value
            new_dimensions = self.dimensions / other.dimensions
            new_unit = f"({self.unit})/({other.unit})"
            
            # Uncertainty propagation for division
            new_uncertainty = None
            if self.uncertainty and other.uncertainty:
                rel_unc_self = self.uncertainty / abs(self.value) if self.value != 0 else 0
                rel_unc_other = other.uncertainty / abs(other.value) if other.value != 0 else 0
                rel_unc_result = np.sqrt(rel_unc_self**2 + rel_unc_other**2)
                new_uncertainty = abs(new_value) * rel_unc_result
            
            return Quantity(new_value, new_unit, new_uncertainty, new_dimensions)
        
        else:
            raise TypeError(f"Cannot divide Quantity by {type(other)}")
    
    def __pow__(self, exponent: Union[float, int]) -> 'Quantity':
        """Raise quantity to a power."""
        if self.value < 0 and not isinstance(exponent, int):
            raise ValueError("Cannot raise negative number to non-integer power")
        
        new_value = self.value ** exponent
        new_dimensions = self.dimensions ** exponent
        new_unit = f"({self.unit})^{exponent}"
        
        # Uncertainty propagation for powers
        new_uncertainty = None
        if self.uncertainty and self.value != 0:
            rel_unc = self.uncertainty / abs(self.value)
            new_uncertainty = abs(new_value) * abs(exponent) * rel_unc
        
        return Quantity(new_value, new_unit, new_uncertainty, new_dimensions)
    
    def __str__(self) -> str:
        """String representation of quantity."""
        if self.uncertainty:
            return f"{self.value:.6g} ± {self.uncertainty:.6g} {self.unit}"
        else:
            return f"{self.value:.6g} {self.unit}"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"Quantity(value={self.value}, unit='{self.unit}', "
                f"uncertainty={self.uncertainty}, dimensions={self.dimensions})")


# Unit definitions with SI base conversions
UNIT_CONVERSIONS = {
    # Length
    'm': 1.0,           # meter (SI base)
    'cm': 0.01,         # centimeter
    'mm': 0.001,        # millimeter
    'μm': 1e-6,         # micrometer
    'um': 1e-6,         # micrometer (ASCII)
    'nm': 1e-9,         # nanometer
    'km': 1000.0,       # kilometer
    'in': 0.0254,       # inch
    'ft': 0.3048,       # foot
    'yd': 0.9144,       # yard
    'mi': 1609.344,     # mile
    'au': 1.495978707e11,  # astronomical unit
    
    # Mass
    'kg': 1.0,          # kilogram (SI base)
    'g': 0.001,         # gram
    'mg': 1e-6,         # milligram
    'lb': 0.45359237,   # pound
    'oz': 0.028349523125,  # ounce
    'ton': 1000.0,      # metric ton
    
    # Time
    's': 1.0,           # second (SI base)
    'ms': 0.001,        # millisecond
    'μs': 1e-6,         # microsecond
    'us': 1e-6,         # microsecond (ASCII)
    'ns': 1e-9,         # nanosecond
    'min': 60.0,        # minute
    'h': 3600.0,        # hour
    'hr': 3600.0,       # hour
    'day': 86400.0,     # day
    'year': 31557600.0, # Julian year
    'yr': 31557600.0,   # year
    
    # Temperature
    'K': 1.0,           # Kelvin (SI base)
    
    # Energy
    'J': 1.0,           # joule (SI derived)
    'kJ': 1000.0,       # kilojoule
    'MJ': 1e6,          # megajoule
    'cal': 4.184,       # calorie
    'kcal': 4184.0,     # kilocalorie
    'eV': 1.602176634e-19,  # electron volt
    'keV': 1.602176634e-16, # kiloelectron volt
    'MeV': 1.602176634e-13, # megaelectron volt
    'Wh': 3600.0,       # watt-hour
    'kWh': 3.6e6,       # kilowatt-hour
    
    # Power
    'W': 1.0,           # watt (SI derived)
    'kW': 1000.0,       # kilowatt
    'MW': 1e6,          # megawatt
    'GW': 1e9,          # gigawatt
    'hp': 745.7,        # horsepower
    
    # Pressure
    'Pa': 1.0,          # pascal (SI derived)
    'kPa': 1000.0,      # kilopascal
    'MPa': 1e6,         # megapascal
    'GPa': 1e9,         # gigapascal
    'bar': 1e5,         # bar
    'atm': 101325.0,    # atmosphere
    'psi': 6894.757,    # pound per square inch
    'Torr': 133.322,    # torr
    'mmHg': 133.322,    # millimeter of mercury
    
    # Angle
    'rad': 1.0,         # radian (SI derived)
    'deg': np.pi/180.0, # degree
    'arcmin': np.pi/10800.0,  # arcminute
    'arcsec': np.pi/648000.0, # arcsecond
    
    # Dimensionless
    '1': 1.0,           # dimensionless
    '': 1.0,            # empty string for dimensionless
}

# Dimensional definitions
UNIT_DIMENSIONS_MAP = {
    # Length [L]
    'm': UnitDimensions(length=1),
    'cm': UnitDimensions(length=1),
    'mm': UnitDimensions(length=1),
    'μm': UnitDimensions(length=1),
    'um': UnitDimensions(length=1),
    'nm': UnitDimensions(length=1),
    'km': UnitDimensions(length=1),
    'in': UnitDimensions(length=1),
    'ft': UnitDimensions(length=1),
    'yd': UnitDimensions(length=1),
    'mi': UnitDimensions(length=1),
    'au': UnitDimensions(length=1),
    
    # Mass [M]
    'kg': UnitDimensions(mass=1),
    'g': UnitDimensions(mass=1),
    'mg': UnitDimensions(mass=1),
    'lb': UnitDimensions(mass=1),
    'oz': UnitDimensions(mass=1),
    'ton': UnitDimensions(mass=1),
    
    # Time [T]
    's': UnitDimensions(time=1),
    'ms': UnitDimensions(time=1),
    'μs': UnitDimensions(time=1),
    'us': UnitDimensions(time=1),
    'ns': UnitDimensions(time=1),
    'min': UnitDimensions(time=1),
    'h': UnitDimensions(time=1),
    'hr': UnitDimensions(time=1),
    'day': UnitDimensions(time=1),
    'year': UnitDimensions(time=1),
    'yr': UnitDimensions(time=1),
    
    # Temperature [Θ]
    'K': UnitDimensions(temperature=1),
    
    # Energy [M L² T⁻²]
    'J': UnitDimensions(mass=1, length=2, time=-2),
    'kJ': UnitDimensions(mass=1, length=2, time=-2),
    'MJ': UnitDimensions(mass=1, length=2, time=-2),
    'cal': UnitDimensions(mass=1, length=2, time=-2),
    'kcal': UnitDimensions(mass=1, length=2, time=-2),
    'eV': UnitDimensions(mass=1, length=2, time=-2),
    'keV': UnitDimensions(mass=1, length=2, time=-2),
    'MeV': UnitDimensions(mass=1, length=2, time=-2),
    'Wh': UnitDimensions(mass=1, length=2, time=-2),
    'kWh': UnitDimensions(mass=1, length=2, time=-2),
    
    # Power [M L² T⁻³]
    'W': UnitDimensions(mass=1, length=2, time=-3),
    'kW': UnitDimensions(mass=1, length=2, time=-3),
    'MW': UnitDimensions(mass=1, length=2, time=-3),
    'GW': UnitDimensions(mass=1, length=2, time=-3),
    'hp': UnitDimensions(mass=1, length=2, time=-3),
    
    # Pressure [M L⁻¹ T⁻²]
    'Pa': UnitDimensions(mass=1, length=-1, time=-2),
    'kPa': UnitDimensions(mass=1, length=-1, time=-2),
    'MPa': UnitDimensions(mass=1, length=-1, time=-2),
    'GPa': UnitDimensions(mass=1, length=-1, time=-2),
    'bar': UnitDimensions(mass=1, length=-1, time=-2),
    'atm': UnitDimensions(mass=1, length=-1, time=-2),
    'psi': UnitDimensions(mass=1, length=-1, time=-2),
    'Torr': UnitDimensions(mass=1, length=-1, time=-2),
    'mmHg': UnitDimensions(mass=1, length=-1, time=-2),
    
    # Angle [A] (dimensionless in SI)
    'rad': UnitDimensions(angle=1),
    'deg': UnitDimensions(angle=1),
    'arcmin': UnitDimensions(angle=1),
    'arcsec': UnitDimensions(angle=1),
    
    # Dimensionless
    '1': UnitDimensions(),
    '': UnitDimensions(),
}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a value from one unit to another.
    
    Args:
        value: Numerical value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
        
    Raises:
        ValueError: If units are not recognized or incompatible
    """
    # Handle empty/dimensionless units
    if not from_unit or from_unit == '1':
        from_unit = ''
    if not to_unit or to_unit == '1':
        to_unit = ''
    
    # Check if units exist
    if from_unit not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown source unit: '{from_unit}'")
    if to_unit not in UNIT_CONVERSIONS:
        raise ValueError(f"Unknown target unit: '{to_unit}'")
    
    # Check dimensional compatibility
    from_dims = get_unit_dimensions(from_unit)
    to_dims = get_unit_dimensions(to_unit)
    if not from_dims == to_dims:
        raise ValueError(
            f"Incompatible units: '{from_unit}' [{from_dims}] "
            f"and '{to_unit}' [{to_dims}]"
        )
    
    # Perform conversion via SI base units
    from_factor = UNIT_CONVERSIONS[from_unit]
    to_factor = UNIT_CONVERSIONS[to_unit]
    
    return value * from_factor / to_factor


def get_unit_dimensions(unit: str) -> UnitDimensions:
    """
    Get the dimensional analysis for a unit.
    
    Args:
        unit: Unit string
        
    Returns:
        UnitDimensions object
        
    Raises:
        ValueError: If unit is not recognized
    """
    if not unit or unit == '1':
        unit = ''
    
    if unit not in UNIT_DIMENSIONS_MAP:
        raise ValueError(f"Unknown unit for dimensional analysis: '{unit}'")
    
    return UNIT_DIMENSIONS_MAP[unit]


def validate_dimensions(quantity1: Quantity, quantity2: Quantity, operation: str) -> bool:
    """
    Validate that two quantities have compatible dimensions for an operation.
    
    Args:
        quantity1: First quantity
        quantity2: Second quantity  
        operation: Type of operation ('add', 'subtract', 'multiply', 'divide')
        
    Returns:
        True if dimensions are compatible
        
    Raises:
        ValueError: If dimensions are incompatible
    """
    if operation in ['add', 'subtract']:
        if not quantity1.dimensions == quantity2.dimensions:
            raise ValueError(
                f"Cannot {operation} quantities with incompatible dimensions: "
                f"{quantity1.dimensions} {operation} {quantity2.dimensions}"
            )
    
    # Multiplication and division are always allowed dimensionally
    return True


def create_quantity(value: float, unit: str, uncertainty: Optional[float] = None) -> Quantity:
    """
    Convenience function to create a Quantity with validation.
    
    Args:
        value: Numerical value
        unit: Unit string
        uncertainty: Optional uncertainty value
        
    Returns:
        Quantity object
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"Value must be numeric, got {type(value)}")
    
    if not isinstance(unit, str):
        raise ValueError(f"Unit must be string, got {type(unit)}")
    
    if uncertainty is not None and uncertainty < 0:
        raise ValueError("Uncertainty must be non-negative")
    
    return Quantity(value, unit, uncertainty)


def temperature_convert(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between different scales.
    
    Special handling for temperature scales that have offset zeros
    (Celsius, Fahrenheit) vs absolute scales (Kelvin, Rankine).
    
    Args:
        value: Temperature value
        from_unit: Source temperature scale ('K', 'C', 'F', 'R')
        to_unit: Target temperature scale
        
    Returns:
        Converted temperature
        
    Raises:
        ValueError: If temperature scales are not recognized
    """
    # First convert to Kelvin
    if from_unit == 'K':
        kelvin_value = value
    elif from_unit == 'C':
        kelvin_value = value + 273.15
    elif from_unit == 'F':
        kelvin_value = (value + 459.67) * 5.0/9.0
    elif from_unit == 'R':  # Rankine
        kelvin_value = value * 5.0/9.0
    else:
        raise ValueError(f"Unknown temperature scale: '{from_unit}'")
    
    # Then convert from Kelvin to target
    if to_unit == 'K':
        return kelvin_value
    elif to_unit == 'C':
        return kelvin_value - 273.15
    elif to_unit == 'F':
        return kelvin_value * 9.0/5.0 - 459.67
    elif to_unit == 'R':  # Rankine
        return kelvin_value * 9.0/5.0
    else:
        raise ValueError(f"Unknown temperature scale: '{to_unit}'")


# Convenience functions for common unit operations
def meters(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create length quantity in meters."""
    return Quantity(value, 'm', uncertainty)

def kilograms(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create mass quantity in kilograms."""
    return Quantity(value, 'kg', uncertainty)

def seconds(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create time quantity in seconds."""
    return Quantity(value, 's', uncertainty)

def kelvin(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create temperature quantity in Kelvin."""
    return Quantity(value, 'K', uncertainty)

def watts(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create power quantity in watts."""
    return Quantity(value, 'W', uncertainty)

def joules(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create energy quantity in joules."""
    return Quantity(value, 'J', uncertainty)

def pascals(value: float, uncertainty: Optional[float] = None) -> Quantity:
    """Create pressure quantity in pascals."""
    return Quantity(value, 'Pa', uncertainty)


# Export main classes and functions
__all__ = [
    'Quantity',
    'UnitDimensions',
    'Dimension',
    'convert_units',
    'get_unit_dimensions',
    'validate_dimensions',
    'create_quantity',
    'temperature_convert',
    'meters', 'kilograms', 'seconds', 'kelvin', 'watts', 'joules', 'pascals'
]
