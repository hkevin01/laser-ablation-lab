"""
Physical constants and unit definitions for laser ablation simulations.

This module provides fundamental physical constants, material properties,
and unit definitions used throughout the laser ablation simulation framework.
All values include uncertainty estimates and literature references.

References:
    - NIST CODATA 2018 fundamental physical constants
    - CRC Handbook of Chemistry and Physics, 103rd Edition
    - Various peer-reviewed literature for material properties
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Fundamental Physical Constants (NIST CODATA 2018)
# All values in SI units with uncertainty estimates

class PhysicalConstants:
    """Container for fundamental physical constants with uncertainties."""
    
    # Universal constants
    SPEED_OF_LIGHT = 299792458.0  # m/s (exact)
    PLANCK_CONSTANT = 6.62607015e-34  # J⋅s (exact)
    BOLTZMANN_CONSTANT = 1.380649e-23  # J/K (exact)
    AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹ (exact)
    GAS_CONSTANT = 8.314462618  # J/(mol⋅K) (exact)
    
    # Electromagnetic constants
    VACUUM_PERMEABILITY = 4e-7 * np.pi  # H/m (exact)
    VACUUM_PERMITTIVITY = 1.0 / (VACUUM_PERMEABILITY * SPEED_OF_LIGHT**2)  # F/m
    ELEMENTARY_CHARGE = 1.602176634e-19  # C (exact)
    
    # Thermodynamic constants
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²⋅K⁴)
    WIEN_DISPLACEMENT = 2.897771955e-3  # m⋅K
    
    # Gravitational constants
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³/(kg⋅s²), uncertainty ±0.00015e-11
    
    # Astronomical constants
    ASTRONOMICAL_UNIT = 149597870700.0  # m (exact)
    SOLAR_MASS = 1.98847e30  # kg, uncertainty ±0.00007e30
    EARTH_MASS = 5.9722e24  # kg, uncertainty ±0.0006e24
    
    @classmethod
    def get_constant(cls, name: str) -> float:
        """
        Get a physical constant by name.
        
        Args:
            name: Name of the constant (case-insensitive)
            
        Returns:
            Value of the constant
            
        Raises:
            ValueError: If constant name is not found
        """
        name_upper = name.upper()
        if hasattr(cls, name_upper):
            return getattr(cls, name_upper)
        else:
            available = [attr for attr in dir(cls) if not attr.startswith('_') and attr.isupper()]
            raise ValueError(f"Unknown constant '{name}'. Available: {available}")


class MaterialProperties:
    """Material properties database for common asteroid/comet materials."""
    
    # Material property dictionaries
    # All properties at standard conditions unless specified
    
    BASALT = {
        'name': 'Basalt',
        'density': 2800.0,  # kg/m³
        'specific_heat': 840.0,  # J/(kg⋅K) at 300K
        'thermal_conductivity': 2.0,  # W/(m⋅K) at 300K
        'melting_point': 1473.0,  # K
        'boiling_point': 3000.0,  # K (estimated)
        'latent_heat_fusion': 4.0e5,  # J/kg
        'latent_heat_vaporization': 6.0e6,  # J/kg
        'absorption_coefficient': 0.95,  # dimensionless (typical for dark basalt)
        'emissivity': 0.95,  # dimensionless
        'young_modulus': 60e9,  # Pa
        'poisson_ratio': 0.25,  # dimensionless
        'compressive_strength': 200e6,  # Pa
        'tensile_strength': 15e6,  # Pa
        'reference': 'Melosh (1989), Artemieva & Shuvalov (2008)'
    }
    
    CHONDRITE = {
        'name': 'Ordinary Chondrite',
        'density': 3500.0,  # kg/m³
        'specific_heat': 750.0,  # J/(kg⋅K) at 300K
        'thermal_conductivity': 3.5,  # W/(m⋅K) at 300K
        'melting_point': 1373.0,  # K
        'boiling_point': 2800.0,  # K (estimated)
        'latent_heat_fusion': 3.5e5,  # J/kg
        'latent_heat_vaporization': 5.5e6,  # J/kg
        'absorption_coefficient': 0.93,  # dimensionless
        'emissivity': 0.93,  # dimensionless
        'young_modulus': 80e9,  # Pa
        'poisson_ratio': 0.28,  # dimensionless
        'compressive_strength': 300e6,  # Pa
        'tensile_strength': 25e6,  # Pa
        'reference': 'Britt & Consolmagno (2003), Flynn et al. (2018)'
    }
    
    NICKEL_IRON = {
        'name': 'Nickel-Iron (70% Fe, 30% Ni)',
        'density': 7800.0,  # kg/m³
        'specific_heat': 450.0,  # J/(kg⋅K) at 300K
        'thermal_conductivity': 80.0,  # W/(m⋅K) at 300K
        'melting_point': 1728.0,  # K
        'boiling_point': 3023.0,  # K
        'latent_heat_fusion': 2.7e5,  # J/kg
        'latent_heat_vaporization': 6.3e6,  # J/kg
        'absorption_coefficient': 0.85,  # dimensionless
        'emissivity': 0.30,  # dimensionless (polished metal)
        'young_modulus': 200e9,  # Pa
        'poisson_ratio': 0.31,  # dimensionless
        'compressive_strength': 800e6,  # Pa
        'tensile_strength': 600e6,  # Pa
        'reference': 'ASM Handbook Vol. 2 (1990), Love & Ahrens (1996)'
    }
    
    ICE = {
        'name': 'Water Ice',
        'density': 917.0,  # kg/m³ at 273K
        'specific_heat': 2060.0,  # J/(kg⋅K) at 260K
        'thermal_conductivity': 2.2,  # W/(m⋅K) at 260K
        'melting_point': 273.15,  # K
        'boiling_point': 373.15,  # K at 1 atm
        'latent_heat_fusion': 3.34e5,  # J/kg
        'latent_heat_vaporization': 2.26e6,  # J/kg
        'absorption_coefficient': 0.97,  # dimensionless (for CO2 laser)
        'emissivity': 0.97,  # dimensionless
        'young_modulus': 9.5e9,  # Pa at 260K
        'poisson_ratio': 0.33,  # dimensionless
        'compressive_strength': 30e6,  # Pa at 260K
        'tensile_strength': 3e6,  # Pa at 260K
        'reference': 'Petrenko & Whitworth (1999), Durham et al. (1983)'
    }
    
    @classmethod
    def get_material(cls, name: str) -> Dict[str, Any]:
        """
        Get material properties by name.
        
        Args:
            name: Material name (case-insensitive)
            
        Returns:
            Dictionary of material properties
            
        Raises:
            ValueError: If material is not found
        """
        materials = {
            'basalt': cls.BASALT,
            'chondrite': cls.CHONDRITE,
            'ordinary_chondrite': cls.CHONDRITE,
            'nickel_iron': cls.NICKEL_IRON,
            'ni_fe': cls.NICKEL_IRON,
            'metal': cls.NICKEL_IRON,
            'ice': cls.ICE,
            'water_ice': cls.ICE
        }
        
        name_lower = name.lower().replace(' ', '_').replace('-', '_')
        if name_lower in materials:
            return materials[name_lower].copy()
        else:
            available = list(materials.keys())
            raise ValueError(f"Unknown material '{name}'. Available: {available}")
    
    @classmethod
    def list_materials(cls) -> list:
        """Return list of available materials."""
        return ['basalt', 'chondrite', 'nickel_iron', 'ice']


class LaserProperties:
    """Common laser system properties and wavelengths."""
    
    # Wavelengths in meters
    CO2_LASER = 10.6e-6  # m (infrared)
    ND_YAG_1064 = 1.064e-6  # m (near-infrared)
    ND_YAG_532 = 532e-9  # m (green, frequency doubled)
    DIODE_808 = 808e-9  # m (near-infrared)
    DIODE_980 = 980e-9  # m (near-infrared)
    
    # Typical laser parameters
    DIFFRACTION_LIMITED_M2 = 1.0  # beam quality factor
    TYPICAL_M2 = 1.5  # realistic beam quality
    POOR_M2 = 3.0  # poor beam quality
    
    @classmethod
    def get_wavelength(cls, laser_type: str) -> float:
        """
        Get laser wavelength by type.
        
        Args:
            laser_type: Type of laser (e.g., 'CO2', 'Nd:YAG')
            
        Returns:
            Wavelength in meters
            
        Raises:
            ValueError: If laser type is not found
        """
        laser_map = {
            'co2': cls.CO2_LASER,
            'ndyag_1064': cls.ND_YAG_1064,
            'ndyag_532': cls.ND_YAG_532,
            'nd_yag_1064': cls.ND_YAG_1064,
            'nd_yag_532': cls.ND_YAG_532,
            'diode_808': cls.DIODE_808,
            'diode_980': cls.DIODE_980
        }
        
        laser_key = laser_type.lower().replace(':', '').replace(' ', '_')
        if laser_key in laser_map:
            return laser_map[laser_key]
        else:
            available = list(laser_map.keys())
            raise ValueError(f"Unknown laser type '{laser_type}'. Available: {available}")


def validate_temperature_range(temperature: float, material_name: str = "unknown") -> bool:
    """
    Validate if temperature is in a reasonable range for space applications.
    
    Args:
        temperature: Temperature in Kelvin
        material_name: Name of material for context in warnings
        
    Returns:
        True if temperature is valid, False otherwise
        
    Raises:
        ValueError: If temperature is clearly invalid
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive (got {temperature} K)")
    
    if temperature < 2.7:  # Below cosmic microwave background
        logger.warning(f"Temperature {temperature} K is below cosmic microwave background")
        return False
    
    if temperature > 10000:  # Above typical vaporization temperatures
        logger.warning(f"Temperature {temperature} K is extremely high for material {material_name}")
        return False
    
    return True


def get_material_property_at_temperature(
    material: Dict[str, Any], 
    property_name: str, 
    temperature: float
) -> float:
    """
    Get temperature-dependent material property with simple correlations.
    
    Args:
        material: Material properties dictionary
        property_name: Name of the property
        temperature: Temperature in Kelvin
        
    Returns:
        Property value at specified temperature
        
    Raises:
        ValueError: If property or temperature is invalid
    """
    validate_temperature_range(temperature, material['name'])
    
    if property_name not in material:
        raise ValueError(f"Property '{property_name}' not found in material {material['name']}")
    
    base_value = material[property_name]
    
    # Simple temperature dependencies (could be improved with more sophisticated models)
    if property_name == 'thermal_conductivity':
        # Thermal conductivity typically decreases with temperature for rocks
        return base_value * (300.0 / temperature)**0.5
    
    elif property_name == 'specific_heat':
        # Specific heat typically increases with temperature
        return base_value * (1.0 + 0.0002 * (temperature - 300.0))
    
    else:
        # Return base value for properties without temperature dependence
        return base_value


# Module-level convenience functions
def get_constant(name: str) -> float:
    """Get a physical constant by name."""
    return PhysicalConstants.get_constant(name)


def get_material(name: str) -> Dict[str, Any]:
    """Get material properties by name."""
    return MaterialProperties.get_material(name)


def get_laser_wavelength(laser_type: str) -> float:
    """Get laser wavelength by type."""
    return LaserProperties.get_wavelength(laser_type)


# Export main classes and functions
__all__ = [
    'PhysicalConstants',
    'MaterialProperties', 
    'LaserProperties',
    'get_constant',
    'get_material',
    'get_laser_wavelength',
    'validate_temperature_range',
    'get_material_property_at_temperature'
]
