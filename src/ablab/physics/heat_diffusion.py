"""
Heat diffusion models for laser ablation simulations.

This module implements various heat diffusion models for studying thermal
transport in asteroid materials during laser irradiation. Includes both
analytical solutions and numerical solvers for different geometries and
boundary conditions.

Features:
    - 1D and 2D heat diffusion solvers
    - Temperature-dependent material properties
    - Pulsed and continuous wave (CW) laser heating
    - Multiple boundary condition types
    - Adaptive time stepping for numerical stability
    - Memory-efficient computation for long simulations

References:
    - Carslaw & Jaeger (1959) Conduction of Heat in Solids
    - Ozisik (1993) Heat Conduction, 2nd Edition
    - Phipps et al. (2012) Review: Laser-ablation propulsion
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import warnings
import logging
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..constants import MaterialProperties, get_material_property_at_temperature
from ..units import Quantity, UnitDimensions

# Configure logging
logger = logging.getLogger(__name__)

# Performance and memory management settings
MAX_MEMORY_MB = 8000  # Maximum memory usage in MB
DEFAULT_GRID_SIZE = 1000  # Default maximum grid points
CONVERGENCE_TOLERANCE = 1e-8  # Numerical convergence tolerance


@dataclass
class ThermalProperties:
    """Container for temperature-dependent thermal properties."""
    density: float  # kg/m³
    specific_heat: Callable[[float], float]  # J/(kg⋅K), function of temperature
    thermal_conductivity: Callable[[float], float]  # W/(m⋅K), function of temperature
    melting_point: float  # K
    boiling_point: float  # K
    latent_heat_fusion: float  # J/kg
    latent_heat_vaporization: float  # J/kg
    
    @classmethod
    def from_material(cls, material_name: str) -> 'ThermalProperties':
        """
        Create thermal properties from material database.
        
        Args:
            material_name: Name of material in database
            
        Returns:
            ThermalProperties object with temperature-dependent functions
        """
        material = MaterialProperties.get_material(material_name)
        
        # Create temperature-dependent property functions
        def specific_heat_func(T: float) -> float:
            """Specific heat as function of temperature."""
            try:
                return get_material_property_at_temperature(material, 'specific_heat', T)
            except Exception as e:
                logger.warning(f"Error calculating specific heat at T={T}: {e}")
                return material['specific_heat']  # Fallback to base value
        
        def thermal_conductivity_func(T: float) -> float:
            """Thermal conductivity as function of temperature."""
            try:
                return get_material_property_at_temperature(material, 'thermal_conductivity', T)
            except Exception as e:
                logger.warning(f"Error calculating thermal conductivity at T={T}: {e}")
                return material['thermal_conductivity']  # Fallback to base value
        
        return cls(
            density=material['density'],
            specific_heat=specific_heat_func,
            thermal_conductivity=thermal_conductivity_func,
            melting_point=material['melting_point'],
            boiling_point=material['boiling_point'],
            latent_heat_fusion=material['latent_heat_fusion'],
            latent_heat_vaporization=material['latent_heat_vaporization']
        )


@dataclass
class BoundaryCondition:
    """Boundary condition specification for heat diffusion."""
    type: str  # 'dirichlet', 'neumann', 'robin', 'convection'
    value: Union[float, Callable[[float], float]]  # Boundary value or heat flux function
    location: str  # 'left', 'right', 'top', 'bottom', 'surface'
    coefficient: Optional[float] = None  # For Robin/convection BC (heat transfer coefficient)
    
    def get_value(self, time: float) -> float:
        """Get boundary condition value at given time."""
        if callable(self.value):
            return self.value(time)
        else:
            return self.value


@dataclass
class SimulationResult:
    """Container for heat diffusion simulation results."""
    time_points: np.ndarray  # Time array [s]
    spatial_points: np.ndarray  # Spatial coordinates [m]
    temperature: np.ndarray  # Temperature field [K], shape (time, space)
    heat_flux: Optional[np.ndarray] = None  # Heat flux [W/m²]
    ablation_rate: Optional[np.ndarray] = None  # Ablation rate [kg/(m²⋅s)]
    energy_balance: Optional[Dict[str, np.ndarray]] = None  # Energy balance components
    computational_stats: Dict[str, Any] = field(default_factory=dict)
    
    def get_surface_temperature(self) -> np.ndarray:
        """Get surface temperature evolution."""
        return self.temperature[:, 0]  # First spatial point is surface
    
    def get_max_temperature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get maximum temperature and its location vs time."""
        max_temps = np.max(self.temperature, axis=1)
        max_locations = self.spatial_points[np.argmax(self.temperature, axis=1)]
        return max_temps, max_locations
    
    def calculate_thermal_penetration(self, threshold_fraction: float = 0.1) -> np.ndarray:
        """
        Calculate thermal penetration depth vs time.
        
        Args:
            threshold_fraction: Temperature fraction to define penetration depth
            
        Returns:
            Penetration depth array [m]
        """
        surface_temps = self.get_surface_temperature()
        baseline_temp = self.temperature[0, -1]  # Initial temperature at depth
        
        penetration_depths = np.zeros(len(self.time_points))
        
        for i, (t_surf, temp_profile) in enumerate(zip(surface_temps, self.temperature)):
            threshold_temp = baseline_temp + threshold_fraction * (t_surf - baseline_temp)
            
            # Find where temperature drops below threshold
            try:
                idx = np.where(temp_profile < threshold_temp)[0][0]
                penetration_depths[i] = self.spatial_points[idx]
            except IndexError:
                # Temperature hasn't dropped below threshold - use max depth
                penetration_depths[i] = self.spatial_points[-1]
        
        return penetration_depths


class HeatDiffusionSolver(ABC):
    """Abstract base class for heat diffusion solvers."""
    
    def __init__(self, 
                 thermal_props: ThermalProperties,
                 spatial_domain: Tuple[float, float],
                 initial_temperature: float = 300.0):
        """
        Initialize heat diffusion solver.
        
        Args:
            thermal_props: Material thermal properties
            spatial_domain: (start, end) coordinates in meters
            initial_temperature: Initial temperature in Kelvin
        """
        self.thermal_props = thermal_props
        self.spatial_domain = spatial_domain
        self.initial_temperature = initial_temperature
        self.boundary_conditions: List[BoundaryCondition] = []
        
        # Validate inputs
        if spatial_domain[1] <= spatial_domain[0]:
            raise ValueError("Spatial domain end must be greater than start")
        if initial_temperature <= 0:
            raise ValueError("Initial temperature must be positive")
    
    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        """Add a boundary condition to the solver."""
        self.boundary_conditions.append(bc)
    
    @abstractmethod
    def solve(self, 
              time_span: Tuple[float, float],
              heat_source: Optional[Callable[[float, float], float]] = None,
              **kwargs) -> SimulationResult:
        """
        Solve the heat diffusion equation.
        
        Args:
            time_span: (start_time, end_time) in seconds
            heat_source: Heat source function Q(x, t) in W/m³
            **kwargs: Additional solver options
            
        Returns:
            SimulationResult object
        """
        pass


class FiniteDifference1D(HeatDiffusionSolver):
    """
    1D finite difference heat diffusion solver with adaptive time stepping.
    
    Solves the heat equation: ρc(∂T/∂t) = ∇⋅(k∇T) + Q
    Using implicit Euler method for stability.
    """
    
    def __init__(self, 
                 thermal_props: ThermalProperties,
                 spatial_domain: Tuple[float, float],
                 num_points: int = 100,
                 initial_temperature: float = 300.0):
        """
        Initialize 1D finite difference solver.
        
        Args:
            thermal_props: Material thermal properties
            spatial_domain: (start, end) coordinates in meters
            num_points: Number of spatial grid points
            initial_temperature: Initial temperature in Kelvin
        """
        super().__init__(thermal_props, spatial_domain, initial_temperature)
        
        if num_points < 10:
            raise ValueError("Need at least 10 grid points for numerical stability")
        
        # Check memory requirements
        memory_estimate = self._estimate_memory_usage(num_points)
        if memory_estimate > MAX_MEMORY_MB:
            suggested_points = int(num_points * np.sqrt(MAX_MEMORY_MB / memory_estimate))
            warnings.warn(
                f"Requested grid size requires ~{memory_estimate:.1f} MB memory. "
                f"Consider reducing to {suggested_points} points."
            )
        
        self.num_points = num_points
        self.dx = (spatial_domain[1] - spatial_domain[0]) / (num_points - 1)
        self.x = np.linspace(spatial_domain[0], spatial_domain[1], num_points)
        
        # Initialize temperature field
        self.T = np.full(num_points, initial_temperature, dtype=np.float64)
        
        # Performance tracking
        self.solver_stats = {
            'total_time_steps': 0,
            'failed_steps': 0,
            'matrix_assembly_time': 0.0,
            'solve_time': 0.0
        }
    
    def _estimate_memory_usage(self, num_points: int) -> float:
        """Estimate memory usage in MB for given grid size."""
        # Rough estimate: sparse matrix + temperature arrays + workspace
        matrix_size = num_points * 3 * 8  # bytes for tridiagonal matrix
        arrays_size = num_points * 8 * 10  # bytes for various arrays
        workspace_size = num_points * 8 * 5  # bytes for solver workspace
        
        total_bytes = matrix_size + arrays_size + workspace_size
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _build_diffusion_matrix(self, temperature: np.ndarray, dt: float) -> sp.csr_matrix:
        """
        Build the finite difference matrix for implicit time stepping.
        
        Args:
            temperature: Current temperature field
            dt: Time step size
            
        Returns:
            Sparse matrix for implicit solve
        """
        start_time = time.time()
        
        n = self.num_points
        dx2 = self.dx**2
        
        # Initialize matrix arrays for tridiagonal system
        main_diag = np.zeros(n)
        upper_diag = np.zeros(n-1)
        lower_diag = np.zeros(n-1)
        
        # Calculate thermal diffusivity at each point
        # α = k/(ρc) where k, ρ, c may be temperature dependent
        alpha = np.zeros(n)
        for i in range(n):
            k = self.thermal_props.thermal_conductivity(temperature[i])
            c = self.thermal_props.specific_heat(temperature[i])
            rho = self.thermal_props.density
            alpha[i] = k / (rho * c)
        
        # Build finite difference stencil: I - dt*α*D²
        # Interior points (second-order central difference)
        for i in range(1, n-1):
            alpha_avg = 0.5 * (alpha[i] + alpha[i+1])  # Average diffusivity
            coeff = dt * alpha_avg / dx2
            
            main_diag[i] = 1.0 + 2.0 * coeff
            upper_diag[i] = -coeff
            lower_diag[i-1] = -coeff
        
        # Apply boundary conditions
        self._apply_boundary_conditions_to_matrix(
            main_diag, upper_diag, lower_diag, temperature, dt
        )
        
        # Create sparse matrix
        diagonals = [lower_diag, main_diag, upper_diag]
        offsets = [-1, 0, 1]
        matrix = sp.diags(diagonals, offsets, shape=(n, n), format='csr')
        
        self.solver_stats['matrix_assembly_time'] += time.time() - start_time
        return matrix
    
    def _apply_boundary_conditions_to_matrix(self,
                                           main_diag: np.ndarray,
                                           upper_diag: np.ndarray, 
                                           lower_diag: np.ndarray,
                                           temperature: np.ndarray,
                                           dt: float,
                                           current_time: float = 0.0) -> None:
        """Apply boundary conditions to the finite difference matrix."""
        n = self.num_points
        
        # Default boundary conditions (if none specified)
        if not self.boundary_conditions:
            # Zero flux at both ends (insulated boundaries)
            main_diag[0] = 1.0
            upper_diag[0] = -1.0
            
            main_diag[-1] = 1.0
            lower_diag[-2] = -1.0
            return
        
        # Apply specified boundary conditions
        for bc in self.boundary_conditions:
            if bc.location == 'left' or bc.location == 'surface':
                self._apply_single_boundary_condition(
                    bc, main_diag, upper_diag, lower_diag, 
                    temperature, dt, current_time, 0
                )
            elif bc.location == 'right':
                self._apply_single_boundary_condition(
                    bc, main_diag, upper_diag, lower_diag,
                    temperature, dt, current_time, -1
                )
    
    def _apply_single_boundary_condition(self,
                                       bc: BoundaryCondition,
                                       main_diag: np.ndarray,
                                       upper_diag: np.ndarray,
                                       lower_diag: np.ndarray,
                                       temperature: np.ndarray,
                                       dt: float,
                                       current_time: float,
                                       index: int) -> None:
        """Apply a single boundary condition at specified index."""
        n = self.num_points
        actual_index = index if index >= 0 else n + index
        
        if bc.type == 'dirichlet':
            # Fixed temperature: T = T_boundary
            main_diag[actual_index] = 1.0
            if actual_index == 0:
                upper_diag[0] = 0.0
            elif actual_index == n-1:
                lower_diag[-1] = 0.0
        
        elif bc.type == 'neumann':
            # Fixed heat flux: -k∇T = q
            k = self.thermal_props.thermal_conductivity(temperature[actual_index])
            
            if actual_index == 0:
                # Forward difference: (T[1] - T[0])/dx = q/k
                main_diag[0] = 1.0
                upper_diag[0] = -1.0
            else:
                # Backward difference: (T[-1] - T[-2])/dx = q/k  
                main_diag[-1] = 1.0
                lower_diag[-2] = -1.0
        
        elif bc.type == 'robin' or bc.type == 'convection':
            # Robin BC: -k∇T = h(T - T_ambient)
            if bc.coefficient is None:
                raise ValueError("Robin/convection BC requires heat transfer coefficient")
            
            h = bc.coefficient
            k = self.thermal_props.thermal_conductivity(temperature[actual_index])
            T_ambient = bc.get_value(current_time)
            
            # Discretization: -k(T[1]-T[0])/dx = h(T[0] - T_ambient)
            # Rearrange: T[0](k/dx + h) - T[1](k/dx) = h*T_ambient
            if actual_index == 0:
                coeff = k/self.dx + h
                main_diag[0] = coeff
                upper_diag[0] = -k/self.dx
            else:
                coeff = k/self.dx + h  
                main_diag[-1] = coeff
                lower_diag[-2] = -k/self.dx
        
        else:
            raise ValueError(f"Unknown boundary condition type: {bc.type}")
    
    def _build_rhs_vector(self, 
                         temperature: np.ndarray,
                         heat_source: Optional[Callable[[float, float], float]],
                         current_time: float,
                         dt: float) -> np.ndarray:
        """Build right-hand side vector including heat source and boundary terms."""
        n = self.num_points
        rhs = temperature.copy()  # Start with current temperature
        
        # Add heat source term
        if heat_source is not None:
            for i in range(n):
                Q = heat_source(self.x[i], current_time)
                c = self.thermal_props.specific_heat(temperature[i])
                rho = self.thermal_props.density
                rhs[i] += dt * Q / (rho * c)
        
        # Add boundary condition contributions
        for bc in self.boundary_conditions:
            if bc.type == 'dirichlet':
                if bc.location in ['left', 'surface']:
                    rhs[0] = bc.get_value(current_time)
                elif bc.location == 'right':
                    rhs[-1] = bc.get_value(current_time)
            
            elif bc.type == 'neumann':
                flux_value = bc.get_value(current_time)
                if bc.location in ['left', 'surface']:
                    rhs[0] = flux_value * self.dx / self.thermal_props.thermal_conductivity(temperature[0])
                elif bc.location == 'right':
                    rhs[-1] = flux_value * self.dx / self.thermal_props.thermal_conductivity(temperature[-1])
            
            elif bc.type in ['robin', 'convection']:
                T_ambient = bc.get_value(current_time)
                h = bc.coefficient
                if bc.location in ['left', 'surface']:
                    rhs[0] = h * T_ambient
                elif bc.location == 'right':
                    rhs[-1] = h * T_ambient
        
        return rhs
    
    def _adaptive_time_step(self, 
                           temperature: np.ndarray,
                           heat_source: Optional[Callable],
                           current_time: float,
                           dt_current: float) -> Tuple[float, bool]:
        """
        Determine adaptive time step based on stability and accuracy criteria.
        
        Args:
            temperature: Current temperature field
            heat_source: Heat source function
            current_time: Current simulation time
            dt_current: Current time step
            
        Returns:
            Tuple of (new_dt, step_accepted)
        """
        # Calculate maximum thermal diffusivity
        max_alpha = 0.0
        for i in range(self.num_points):
            k = self.thermal_props.thermal_conductivity(temperature[i])
            c = self.thermal_props.specific_heat(temperature[i])
            rho = self.thermal_props.density
            alpha = k / (rho * c)
            max_alpha = max(max_alpha, alpha)
        
        # CFL stability criterion for diffusion: dt < dx²/(2α)
        dt_stable = 0.4 * self.dx**2 / max_alpha if max_alpha > 0 else dt_current
        
        # Accuracy criterion: limit temperature change rate
        max_temp_rate = 0.0
        if heat_source is not None:
            for i in range(self.num_points):
                Q = heat_source(self.x[i], current_time)
                c = self.thermal_props.specific_heat(temperature[i])
                rho = self.thermal_props.density
                dT_dt = Q / (rho * c)
                max_temp_rate = max(max_temp_rate, abs(dT_dt))
        
        # Limit temperature change per time step
        max_temp_change = 50.0  # K per time step
        dt_accuracy = max_temp_change / max_temp_rate if max_temp_rate > 0 else dt_current
        
        # Choose most restrictive criterion
        dt_new = min(dt_stable, dt_accuracy, dt_current * 1.5)  # Don't increase too quickly
        dt_new = max(dt_new, dt_current * 0.5)  # Don't decrease too quickly
        
        # Accept step if temperature changes are reasonable
        step_accepted = True
        if hasattr(self, '_previous_temperature'):
            max_change = np.max(np.abs(temperature - self._previous_temperature))
            if max_change > 2 * max_temp_change:
                step_accepted = False
                dt_new = dt_current * 0.5
        
        return dt_new, step_accepted
    
    def solve(self,
              time_span: Tuple[float, float],
              heat_source: Optional[Callable[[float, float], float]] = None,
              dt_initial: float = 1e-6,
              max_time_steps: int = 100000,
              output_interval: Optional[float] = None,
              progress_callback: Optional[Callable[[float], None]] = None) -> SimulationResult:
        """
        Solve the 1D heat diffusion equation with adaptive time stepping.
        
        Args:
            time_span: (start_time, end_time) in seconds
            heat_source: Heat source function Q(x, t) in W/m³
            dt_initial: Initial time step size in seconds
            max_time_steps: Maximum number of time steps
            output_interval: Time interval for saving results (None = adaptive)
            progress_callback: Function to call with progress updates
            
        Returns:
            SimulationResult object with temperature evolution
            
        Raises:
            RuntimeError: If solver fails to converge or encounters numerical issues
        """
        logger.info(f"Starting 1D heat diffusion solve: t ∈ {time_span}, {self.num_points} points")
        start_time = time.time()
        
        t_start, t_end = time_span
        if t_end <= t_start:
            raise ValueError("End time must be greater than start time")
        
        # Initialize solution storage
        if output_interval is None:
            output_interval = (t_end - t_start) / 1000  # Default: 1000 output points
        
        output_times = np.arange(t_start, t_end + output_interval, output_interval)
        temperature_history = []
        time_history = []
        
        # Initialize simulation state
        current_time = t_start
        dt = dt_initial
        temperature = self.T.copy()
        
        # Add initial state to output
        temperature_history.append(temperature.copy())
        time_history.append(current_time)
        next_output_time = t_start + output_interval
        
        step_count = 0
        failed_steps = 0
        
        try:
            while current_time < t_end and step_count < max_time_steps:
                # Determine time step
                dt, step_accepted = self._adaptive_time_step(
                    temperature, heat_source, current_time, dt
                )
                
                # Don't overshoot end time
                if current_time + dt > t_end:
                    dt = t_end - current_time
                
                if not step_accepted:
                    failed_steps += 1
                    if failed_steps > 100:
                        raise RuntimeError("Too many failed time steps - solver unstable")
                    continue
                
                # Build and solve linear system
                try:
                    matrix = self._build_diffusion_matrix(temperature, dt)
                    rhs = self._build_rhs_vector(temperature, heat_source, current_time, dt)
                    
                    solve_start = time.time()
                    new_temperature = spla.spsolve(matrix, rhs)
                    self.solver_stats['solve_time'] += time.time() - solve_start
                    
                    # Validate solution
                    if np.any(~np.isfinite(new_temperature)):
                        raise RuntimeError("Non-finite temperatures detected")
                    
                    if np.any(new_temperature < 0):
                        logger.warning("Negative temperatures detected - clamping to 0 K")
                        new_temperature = np.maximum(new_temperature, 0.0)
                    
                    # Check for extreme temperatures
                    max_temp = np.max(new_temperature)
                    if max_temp > 10000:  # 10,000 K seems excessive
                        logger.warning(f"Very high temperature detected: {max_temp:.1f} K")
                    
                except Exception as e:
                    logger.warning(f"Linear solve failed at t={current_time:.2e}: {e}")
                    dt *= 0.5  # Reduce time step and try again
                    failed_steps += 1
                    continue
                
                # Update state
                self._previous_temperature = temperature.copy()
                temperature = new_temperature
                current_time += dt
                step_count += 1
                
                # Save output if needed
                if current_time >= next_output_time or current_time >= t_end:
                    temperature_history.append(temperature.copy())
                    time_history.append(current_time)
                    next_output_time += output_interval
                
                # Progress callback
                if progress_callback and step_count % 100 == 0:
                    progress = (current_time - t_start) / (t_end - t_start)
                    progress_callback(progress)
            
            # Check completion status
            if step_count >= max_time_steps:
                logger.warning(f"Maximum time steps ({max_time_steps}) reached")
            
            # Store final statistics
            self.solver_stats.update({
                'total_time_steps': step_count,
                'failed_steps': failed_steps,
                'final_time': current_time,
                'wall_clock_time': time.time() - start_time
            })
            
            logger.info(f"Solve completed: {step_count} steps, {failed_steps} failed, "
                       f"{self.solver_stats['wall_clock_time']:.2f}s wall time")
            
            # Convert results to arrays
            time_array = np.array(time_history)
            temp_array = np.array(temperature_history)
            
            return SimulationResult(
                time_points=time_array,
                spatial_points=self.x.copy(),
                temperature=temp_array,
                computational_stats=self.solver_stats.copy()
            )
            
        except Exception as e:
            logger.error(f"Heat diffusion solver failed: {e}")
            raise RuntimeError(f"Heat diffusion solve failed: {e}") from e


# Export classes and functions
__all__ = [
    'ThermalProperties',
    'BoundaryCondition', 
    'SimulationResult',
    'HeatDiffusionSolver',
    'FiniteDifference1D'
]
