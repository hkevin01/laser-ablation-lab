"""
Tests for constants module.
"""

import pytest
import numpy as np
from ablab.constants import (
    PhysicalConstants, MaterialProperties, LaserProperties,
    get_constant, get_material, get_laser_wavelength,
    validate_temperature_range, get_material_property_at_temperature
)


class TestPhysicalConstants:
    """Test physical constants functionality."""
    
    def test_speed_of_light(self):
        """Test speed of light constant."""
        c = PhysicalConstants.SPEED_OF_LIGHT
        assert c == 299792458.0
        assert isinstance(c, float)
    
    def test_get_constant(self):
        """Test getting constants by name."""
        c = PhysicalConstants.get_constant("SPEED_OF_LIGHT")
        assert c == 299792458.0
        
        # Test case insensitive
        c2 = PhysicalConstants.get_constant("speed_of_light")
        assert c2 == c
    
    def test_get_constant_invalid(self):
        """Test error handling for invalid constant names."""
        with pytest.raises(ValueError, match="Unknown constant"):
            PhysicalConstants.get_constant("INVALID_CONSTANT")


class TestMaterialProperties:
    """Test material properties functionality."""
    
    def test_get_basalt(self):
        """Test getting basalt properties."""
        basalt = MaterialProperties.get_material("basalt")
        assert basalt["name"] == "Basalt"
        assert basalt["density"] == 2800.0
        assert basalt["melting_point"] == 1473.0
        assert "reference" in basalt
    
    def test_get_material_case_insensitive(self):
        """Test case insensitive material lookup."""
        basalt1 = MaterialProperties.get_material("basalt")
        basalt2 = MaterialProperties.get_material("BASALT")
        basalt3 = MaterialProperties.get_material("Basalt")
        
        assert basalt1["name"] == basalt2["name"] == basalt3["name"]
    
    def test_get_material_aliases(self):
        """Test material name aliases."""
        chondrite1 = MaterialProperties.get_material("chondrite")
        chondrite2 = MaterialProperties.get_material("ordinary_chondrite")
        
        assert chondrite1["name"] == chondrite2["name"]
    
    def test_get_material_invalid(self):
        """Test error handling for invalid material names."""
        with pytest.raises(ValueError, match="Unknown material"):
            MaterialProperties.get_material("invalid_material")
    
    def test_list_materials(self):
        """Test listing available materials."""
        materials = MaterialProperties.list_materials()
        assert isinstance(materials, list)
        assert "basalt" in materials
        assert "chondrite" in materials
        assert "nickel_iron" in materials
        assert "ice" in materials


class TestLaserProperties:
    """Test laser properties functionality."""
    
    def test_co2_wavelength(self):
        """Test CO2 laser wavelength."""
        wavelength = LaserProperties.CO2_LASER
        assert wavelength == 10.6e-6
    
    def test_get_wavelength(self):
        """Test getting wavelength by laser type."""
        co2_wl = LaserProperties.get_wavelength("CO2")
        assert co2_wl == 10.6e-6
        
        ndyag_wl = LaserProperties.get_wavelength("Nd:YAG_1064")
        assert ndyag_wl == 1.064e-6
    
    def test_get_wavelength_invalid(self):
        """Test error handling for invalid laser types."""
        with pytest.raises(ValueError, match="Unknown laser type"):
            LaserProperties.get_wavelength("invalid_laser")


class TestTemperatureValidation:
    """Test temperature validation functionality."""
    
    def test_valid_temperatures(self):
        """Test validation of valid temperatures."""
        assert validate_temperature_range(300.0) is True
        assert validate_temperature_range(1000.0) is True
        assert validate_temperature_range(3000.0) is True
    
    def test_invalid_temperatures(self):
        """Test validation of invalid temperatures."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            validate_temperature_range(-10.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            validate_temperature_range(0.0)
    
    def test_extreme_temperatures(self):
        """Test validation of extreme temperatures."""
        # Very low temperature (should warn but return False)
        assert validate_temperature_range(1.0) is False
        
        # Very high temperature (should warn but return False)
        assert validate_temperature_range(15000.0) is False


class TestTemperatureDependentProperties:
    """Test temperature-dependent material properties."""
    
    def test_thermal_conductivity_temperature_dependence(self):
        """Test thermal conductivity temperature dependence."""
        basalt = get_material("basalt")
        
        k_300 = get_material_property_at_temperature(basalt, "thermal_conductivity", 300.0)
        k_600 = get_material_property_at_temperature(basalt, "thermal_conductivity", 600.0)
        
        # Thermal conductivity should decrease with temperature for rocks
        assert k_600 < k_300
        assert k_300 > 0
        assert k_600 > 0
    
    def test_specific_heat_temperature_dependence(self):
        """Test specific heat temperature dependence."""
        basalt = get_material("basalt")
        
        c_300 = get_material_property_at_temperature(basalt, "specific_heat", 300.0)
        c_600 = get_material_property_at_temperature(basalt, "specific_heat", 600.0)
        
        # Specific heat should increase with temperature
        assert c_600 > c_300
        assert c_300 > 0
        assert c_600 > 0
    
    def test_property_not_found(self):
        """Test error handling for non-existent properties."""
        basalt = get_material("basalt")
        
        with pytest.raises(ValueError, match="Property .* not found"):
            get_material_property_at_temperature(basalt, "invalid_property", 300.0)
    
    def test_invalid_temperature(self):
        """Test error handling for invalid temperatures."""
        basalt = get_material("basalt")
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            get_material_property_at_temperature(basalt, "density", -100.0)


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_constant_function(self):
        """Test get_constant convenience function."""
        c = get_constant("SPEED_OF_LIGHT")
        assert c == PhysicalConstants.SPEED_OF_LIGHT
    
    def test_get_material_function(self):
        """Test get_material convenience function."""
        basalt = get_material("basalt")
        assert basalt["name"] == "Basalt"
    
    def test_get_laser_wavelength_function(self):
        """Test get_laser_wavelength convenience function."""
        wl = get_laser_wavelength("CO2")
        assert wl == LaserProperties.CO2_LASER


@pytest.mark.physics
class TestPhysicsValidation:
    """Physics validation tests against known values."""
    
    def test_material_property_ranges(self):
        """Test that material properties are in reasonable ranges."""
        for material_name in ["basalt", "chondrite", "nickel_iron", "ice"]:
            material = get_material(material_name)
            
            # Density should be reasonable for space materials
            assert 500 < material["density"] < 20000  # kg/m³
            
            # Melting point should be positive and reasonable
            assert 0 < material["melting_point"] < 5000  # K
            
            # Specific heat should be positive
            assert material["specific_heat"] > 0  # J/(kg·K)
            
            # Thermal conductivity should be positive
            assert material["thermal_conductivity"] > 0  # W/(m·K)
    
    def test_physical_constant_values(self):
        """Test that physical constants have correct values."""
        # Test some well-known constants
        assert abs(PhysicalConstants.SPEED_OF_LIGHT - 299792458.0) < 1e-6
        assert abs(PhysicalConstants.BOLTZMANN_CONSTANT - 1.380649e-23) < 1e-28
        assert abs(PhysicalConstants.STEFAN_BOLTZMANN - 5.670374419e-8) < 1e-14
