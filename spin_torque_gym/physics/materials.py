"""Material parameter database for spintronic devices.

This module provides experimentally validated material parameters for common
spintronic materials and interfaces, including temperature and field dependence.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MaterialProperties:
    """Container for material properties."""
    name: str
    saturation_magnetization: float  # A/m
    exchange_constant: float  # J/m
    gilbert_damping: float  # dimensionless
    uniaxial_anisotropy: float  # J/m³
    g_factor: float  # dimensionless
    curie_temperature: float  # K
    density: float  # kg/m³
    resistivity: float  # Ω·m
    spin_polarization: float  # dimensionless

    # Temperature dependence parameters
    ms_temperature_coeff: float = 0.0  # 1/K
    damping_temperature_coeff: float = 0.0  # 1/K
    anisotropy_temperature_coeff: float = 0.0  # 1/K


class MaterialDatabase:
    """Database of spintronic material properties."""

    def __init__(self, custom_materials: Optional[Dict[str, MaterialProperties]] = None):
        """Initialize material database.
        
        Args:
            custom_materials: Optional custom material definitions
        """
        self._materials = self._initialize_default_materials()

        if custom_materials:
            self._materials.update(custom_materials)

    def _initialize_default_materials(self) -> Dict[str, MaterialProperties]:
        """Initialize database with common spintronic materials."""
        materials = {}

        # CoFeB - Common STT-MRAM free layer
        materials['CoFeB'] = MaterialProperties(
            name='CoFeB',
            saturation_magnetization=800e3,  # A/m
            exchange_constant=20e-12,  # J/m
            gilbert_damping=0.01,
            uniaxial_anisotropy=1.0e6,  # J/m³
            g_factor=2.1,
            curie_temperature=650,  # K
            density=7800,  # kg/m³
            resistivity=150e-8,  # Ω·m
            spin_polarization=0.7,
            ms_temperature_coeff=-2e-3,  # 1/K
            damping_temperature_coeff=1e-5,  # 1/K
            anisotropy_temperature_coeff=-3e3  # J/m³/K
        )

        # Fe - Iron reference
        materials['Fe'] = MaterialProperties(
            name='Fe',
            saturation_magnetization=1.7e6,  # A/m
            exchange_constant=21e-12,  # J/m
            gilbert_damping=0.002,
            uniaxial_anisotropy=0.5e6,  # J/m³
            g_factor=2.09,
            curie_temperature=1043,  # K
            density=7870,  # kg/m³
            resistivity=10e-8,  # Ω·m
            spin_polarization=0.44,
            ms_temperature_coeff=-1.5e-3,  # 1/K
            damping_temperature_coeff=5e-6,  # 1/K
            anisotropy_temperature_coeff=-1e3  # J/m³/K
        )

        # Co - Cobalt
        materials['Co'] = MaterialProperties(
            name='Co',
            saturation_magnetization=1.4e6,  # A/m
            exchange_constant=30e-12,  # J/m
            gilbert_damping=0.005,
            uniaxial_anisotropy=4.5e5,  # J/m³
            g_factor=2.18,
            curie_temperature=1388,  # K
            density=8900,  # kg/m³
            resistivity=6e-8,  # Ω·m
            spin_polarization=0.34,
            ms_temperature_coeff=-1.2e-3,  # 1/K
            damping_temperature_coeff=8e-6,  # 1/K
            anisotropy_temperature_coeff=-2e3  # J/m³/K
        )

        # Ni - Nickel
        materials['Ni'] = MaterialProperties(
            name='Ni',
            saturation_magnetization=485e3,  # A/m
            exchange_constant=9e-12,  # J/m
            gilbert_damping=0.045,
            uniaxial_anisotropy=-0.5e5,  # J/m³ (easy plane)
            g_factor=2.18,
            curie_temperature=627,  # K
            density=8900,  # kg/m³
            resistivity=7e-8,  # Ω·m
            spin_polarization=0.11,
            ms_temperature_coeff=-2.5e-3,  # 1/K
            damping_temperature_coeff=2e-5,  # 1/K
            anisotropy_temperature_coeff=-1e2  # J/m³/K
        )

        # Pt - Platinum (SOT applications)
        materials['Pt'] = MaterialProperties(
            name='Pt',
            saturation_magnetization=0,  # Non-magnetic
            exchange_constant=0,
            gilbert_damping=0,
            uniaxial_anisotropy=0,
            g_factor=0,
            curie_temperature=0,
            density=21450,  # kg/m³
            resistivity=10.6e-8,  # Ω·m
            spin_polarization=0,
            ms_temperature_coeff=0,
            damping_temperature_coeff=0,
            anisotropy_temperature_coeff=0
        )

        # Ta - Tantalum (SOT applications)
        materials['Ta'] = MaterialProperties(
            name='Ta',
            saturation_magnetization=0,  # Non-magnetic
            exchange_constant=0,
            gilbert_damping=0,
            uniaxial_anisotropy=0,
            g_factor=0,
            curie_temperature=0,
            density=16650,  # kg/m³
            resistivity=12.4e-8,  # Ω·m
            spin_polarization=0,
            ms_temperature_coeff=0,
            damping_temperature_coeff=0,
            anisotropy_temperature_coeff=0
        )

        # W - Tungsten (SOT applications)
        materials['W'] = MaterialProperties(
            name='W',
            saturation_magnetization=0,  # Non-magnetic
            exchange_constant=0,
            gilbert_damping=0,
            uniaxial_anisotropy=0,
            g_factor=0,
            curie_temperature=0,
            density=19300,  # kg/m³
            resistivity=5.6e-8,  # Ω·m
            spin_polarization=0,
            ms_temperature_coeff=0,
            damping_temperature_coeff=0,
            anisotropy_temperature_coeff=0
        )

        return materials

    def get_material(self, name: str) -> MaterialProperties:
        """Get material properties by name.
        
        Args:
            name: Material name
            
        Returns:
            MaterialProperties object
            
        Raises:
            KeyError: If material not found
        """
        if name not in self._materials:
            available = list(self._materials.keys())
            raise KeyError(f"Material '{name}' not found. Available: {available}")

        return self._materials[name]

    def list_materials(self) -> List[str]:
        """List all available materials."""
        return list(self._materials.keys())

    def add_material(self, material: MaterialProperties) -> None:
        """Add custom material to database."""
        self._materials[material.name] = material

    def get_temperature_adjusted_properties(
        self,
        material_name: str,
        temperature: float
    ) -> Dict[str, float]:
        """Get temperature-adjusted material properties.
        
        Args:
            material_name: Name of material
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary with temperature-adjusted properties
        """
        material = self.get_material(material_name)
        reference_temp = 300.0  # K
        delta_t = temperature - reference_temp

        # Temperature-adjusted properties
        ms_adjusted = material.saturation_magnetization * (
            1 + material.ms_temperature_coeff * delta_t
        )

        damping_adjusted = material.gilbert_damping * (
            1 + material.damping_temperature_coeff * delta_t
        )

        anisotropy_adjusted = material.uniaxial_anisotropy + (
            material.anisotropy_temperature_coeff * delta_t
        )

        return {
            'saturation_magnetization': max(0, ms_adjusted),
            'gilbert_damping': max(0, damping_adjusted),
            'uniaxial_anisotropy': anisotropy_adjusted,
            'exchange_constant': material.exchange_constant,
            'spin_polarization': material.spin_polarization,
            'g_factor': material.g_factor,
            'density': material.density,
            'resistivity': material.resistivity
        }

    def create_bilayer_properties(
        self,
        layer1_name: str,
        layer1_thickness: float,
        layer2_name: str,
        layer2_thickness: float,
        interface_coupling: float = 0.0
    ) -> Dict[str, float]:
        """Create effective properties for bilayer system.
        
        Args:
            layer1_name: Name of first layer material
            layer1_thickness: Thickness of first layer (m)
            layer2_name: Name of second layer material
            layer2_thickness: Thickness of second layer (m)
            interface_coupling: Interface exchange coupling (J/m²)
            
        Returns:
            Dictionary with effective bilayer properties
        """
        mat1 = self.get_material(layer1_name)
        mat2 = self.get_material(layer2_name)

        total_thickness = layer1_thickness + layer2_thickness

        # Volume-weighted average properties
        w1 = layer1_thickness / total_thickness
        w2 = layer2_thickness / total_thickness

        effective_props = {
            'saturation_magnetization': (
                w1 * mat1.saturation_magnetization +
                w2 * mat2.saturation_magnetization
            ),
            'gilbert_damping': (
                w1 * mat1.gilbert_damping +
                w2 * mat2.gilbert_damping
            ),
            'exchange_constant': (
                w1 * mat1.exchange_constant +
                w2 * mat2.exchange_constant
            ),
            'uniaxial_anisotropy': (
                w1 * mat1.uniaxial_anisotropy +
                w2 * mat2.uniaxial_anisotropy
            ),
            'spin_polarization': (
                w1 * mat1.spin_polarization +
                w2 * mat2.spin_polarization
            ),
            'density': (
                w1 * mat1.density +
                w2 * mat2.density
            ),
            'interface_coupling': interface_coupling,
            'total_thickness': total_thickness
        }

        return effective_props

    def export_to_json(self, filename: str) -> None:
        """Export material database to JSON file.
        
        Args:
            filename: Output filename
        """
        export_data = {}

        for name, material in self._materials.items():
            export_data[name] = {
                'name': material.name,
                'saturation_magnetization': material.saturation_magnetization,
                'exchange_constant': material.exchange_constant,
                'gilbert_damping': material.gilbert_damping,
                'uniaxial_anisotropy': material.uniaxial_anisotropy,
                'g_factor': material.g_factor,
                'curie_temperature': material.curie_temperature,
                'density': material.density,
                'resistivity': material.resistivity,
                'spin_polarization': material.spin_polarization,
                'ms_temperature_coeff': material.ms_temperature_coeff,
                'damping_temperature_coeff': material.damping_temperature_coeff,
                'anisotropy_temperature_coeff': material.anisotropy_temperature_coeff
            }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

    def load_from_json(self, filename: str) -> None:
        """Load materials from JSON file.
        
        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        for name, props in data.items():
            material = MaterialProperties(**props)
            self._materials[name] = material

    def find_materials_by_property(
        self,
        property_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> List[str]:
        """Find materials with properties in specified range.
        
        Args:
            property_name: Property to search by
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            List of material names matching criteria
        """
        matching_materials = []

        for name, material in self._materials.items():
            if not hasattr(material, property_name):
                continue

            value = getattr(material, property_name)

            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue

            matching_materials.append(name)

        return matching_materials

    def get_recommended_parameters(
        self,
        device_type: str,
        temperature: float = 300.0
    ) -> Dict[str, Any]:
        """Get recommended material parameters for device type.
        
        Args:
            device_type: Type of device ('stt_mram', 'sot_mram', 'vcma_mram')
            temperature: Operating temperature (K)
            
        Returns:
            Dictionary with recommended parameters
        """
        recommendations = {
            'stt_mram': {
                'free_layer': 'CoFeB',
                'reference_layer': 'CoFeB',
                'barrier': 'MgO',
                'typical_thickness': 1.5e-9,  # m
                'target_thermal_stability': 60
            },
            'sot_mram': {
                'free_layer': 'CoFeB',
                'heavy_metal': 'Pt',
                'barrier': 'MgO',
                'typical_thickness': 1.0e-9,  # m
                'target_thermal_stability': 40
            },
            'vcma_mram': {
                'free_layer': 'CoFeB',
                'barrier': 'MgO',
                'typical_thickness': 1.2e-9,  # m
                'target_thermal_stability': 50
            }
        }

        if device_type not in recommendations:
            available = list(recommendations.keys())
            raise ValueError(f"Unknown device type '{device_type}'. Available: {available}")

        config = recommendations[device_type].copy()

        # Add temperature-adjusted properties for main magnetic layer
        main_material = config.get('free_layer', 'CoFeB')
        temp_props = self.get_temperature_adjusted_properties(main_material, temperature)
        config.update(temp_props)

        return config
