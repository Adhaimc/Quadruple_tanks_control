"""Main simulator class for quadruple tanks system."""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from ..models import QuadrupleTanksSystem


class Simulator:
    """
    Simulator for the quadruple tanks system with control.
    
    This class integrates the plant model with controllers and
    manages the simulation loop.
    """
    
    def __init__(self, 
                 system: QuadrupleTanksSystem = None,
                 controller_pump1: Optional[Any] = None,
                 controller_pump2: Optional[Any] = None,
                 controller_13: Optional[Any] = None,
                 controller_24: Optional[Any] = None,
                 controller_3: Optional[Any] = None,
                 controller_4: Optional[Any] = None,
                 dt: float = 0.1):
        """
        Initialize simulator with PUMP-BASED control (new architecture).
        
        Args:
            system: QuadrupleTanksSystem instance
            controller_pump1: Controller for pump 1 flow (regulates Tank 1)
            controller_pump2: Controller for pump 2 flow (regulates Tank 2)
            controller_13, controller_24, controller_3, controller_4: (Deprecated, kept for compatibility)
            dt: Simulation time step [s]
        """
        self.system = system if system is not None else QuadrupleTanksSystem()
        
        # PUMP CONTROL: Two pump controllers regulate upper tanks
        self.controller_pump1 = controller_pump1
        self.controller_pump2 = controller_pump2
        
        # (Legacy valve controllers, deprecated)
        self.controller_13 = controller_13
        self.controller_24 = controller_24
        self.controller_3 = controller_3
        self.controller_4 = controller_4
        
        self.dt = dt
        
        # Simulation state
        self.current_time = 0.0
        self.setpoint1 = 75.0  # Upper tank 1 setpoint
        self.setpoint2 = 75.0  # Upper tank 2 setpoint
        self.setpoint3 = 75.0  # Lower tank 3 setpoint (passive, follows coupling)
        self.setpoint4 = 75.0  # Lower tank 4 setpoint (passive, follows coupling)
        
        # Data logging
        self.time_data = []
        self.heights_data = {1: [], 2: [], 3: [], 4: []}
        self.pump_data = {"pump1_flow": [], "pump2_flow": []}  # Changed from valve_data
        self.setpoint_data = {"tank1": [], "tank2": [], "tank3": [], "tank4": []}
        
    def set_setpoints(self, sp1: float, sp2: float, sp3: float = 75.0, sp4: float = 75.0) -> None:
        """
        Set desired setpoints.
        
        Note: In pump-based control, only sp1 and sp2 are actively controlled.
        sp3 and sp4 are informational (lower tanks follow passively).
        """
        self.setpoint1 = sp1  # ACTIVE: controlled by pump 1
        self.setpoint2 = sp2  # ACTIVE: controlled by pump 2
        self.setpoint3 = sp3  # PASSIVE: lower tank 3, follows coupling
        self.setpoint4 = sp4  # PASSIVE: lower tank 4, follows coupling
    
    def step(self) -> Dict:
        """Execute one simulation step using MULTI-INPUT control."""
        pump1_flow = 250.0
        pump2_flow = 250.0
        valve3_opening = 0.5
        valve4_opening = 0.5
        
        # Get current tank heights
        heights = self.system.get_heights()
        
        # MULTI-INPUT CONTROL: Controller handles all 4 inputs (u1, u2, u3, u4)
        if self.controller_pump1 is not None:
            # Call the MultiplexController with all 4 setpoints and measurements
            pump1_flow, pump2_flow, valve3_opening, valve4_opening = self.controller_pump1.update(
                self.setpoint1, heights[1],  # Tank 1
                self.setpoint2, heights[2],  # Tank 2
                self.setpoint3, heights[3],  # Tank 3
                self.setpoint4, heights[4],  # Tank 4
                self.current_time
            )
        
        # Apply control commands to system
        self.system.set_pump_flows(pump1_flow, pump2_flow)
        self.system.set_drain_valves(valve3_opening, valve4_opening)
        
        # Update system state
        self.system.update(self.dt)
        
        # Record data
        self._record_data()
        
        # Update time
        self.current_time += self.dt
        
        return self.system.get_state()
    
    def _record_data(self) -> None:
        """Record current simulation data."""
        self.time_data.append(self.current_time)
        
        heights = self.system.get_heights()
        for tank_id in range(1, 5):
            self.heights_data[tank_id].append(heights[tank_id])
        
        # Log pump flows instead of valve positions
        self.pump_data["pump1_flow"].append(self.system.pump1_flow_command)
        self.pump_data["pump2_flow"].append(self.system.pump2_flow_command)
        
        self.setpoint_data["tank1"].append(self.setpoint1)
        self.setpoint_data["tank2"].append(self.setpoint2)
        self.setpoint_data["tank3"].append(self.setpoint3)
        self.setpoint_data["tank4"].append(self.setpoint4)
    
    def run(self, duration: float, setpoint1: float = None, 
            setpoint2: float = None, setpoint3: float = None, setpoint4: float = None) -> Tuple[List, Dict]:
        """Run simulation for specified duration."""
        if setpoint1 is not None: self.setpoint1 = setpoint1
        if setpoint2 is not None: self.setpoint2 = setpoint2
        if setpoint3 is not None: self.setpoint3 = setpoint3
        if setpoint4 is not None: self.setpoint4 = setpoint4
        
        num_steps = int(duration / self.dt)
        
        for _ in range(num_steps):
            self.step()
        
        return self._get_simulation_data()
    
    def _get_simulation_data(self) -> Tuple[List, Dict]:
        """Get all recorded simulation data."""
        data = {
            "heights": self.heights_data,
            "pumps": self.pump_data,  # Changed from valves to pumps
            "setpoints": self.setpoint_data,
        }
        return np.array(self.time_data), data
    
    def reset(self, initial_heights: Dict[int, float] = None) -> None:
        """Reset simulator to initial state."""
        self.system.reset(initial_heights)
        self.current_time = 0.0
        
        # Reset pump controllers
        if self.controller_pump1 is not None: self.controller_pump1.reset()
        if self.controller_pump2 is not None: self.controller_pump2.reset()
        
        # Reset legacy valve controllers (for compatibility)
        if self.controller_13 is not None: self.controller_13.reset()
        if self.controller_24 is not None: self.controller_24.reset()
        if self.controller_3 is not None: self.controller_3.reset()
        if self.controller_4 is not None: self.controller_4.reset()
        
        self.time_data = []
        self.heights_data = {1: [], 2: [], 3: [], 4: []}
        self.pump_data = {"pump1_flow": [], "pump2_flow": []}
        self.setpoint_data = {"tank1": [], "tank2": [], "tank3": [], "tank4": []}
    
    def get_current_state(self) -> Dict:
        """
        Get current system state.
        
        Returns:
            Dictionary with current state variables
        """
        return {
            "time": self.current_time,
            "heights": self.system.get_heights(),
            "pumps": {
                "pump1": self.system.pump1_flow_command,
                "pump2": self.system.pump2_flow_command,
            },
            "drain_valves": {
                "u3": self.system.u3,
                "u4": self.system.u4,
            },
        }
    
    def export_data(self, filename: str) -> None:
        """
        Export simulation data to CSV file.
        
        Args:
            filename: Output filename
        """
        import csv
        
        time_data, state_data = self._get_simulation_data()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['Time', 'H1', 'H2', 'H3', 'H4', 'Pump1_Flow', 'Pump2_Flow', 'SP1', 'SP2', 'SP3', 'SP4']
            writer.writerow(header)
            
            # Write data rows
            for i, t in enumerate(time_data):
                row = [
                    t,
                    state_data["heights"][1][i],
                    state_data["heights"][2][i],
                    state_data["heights"][3][i],
                    state_data["heights"][4][i],
                    state_data["pumps"]["pump1_flow"][i],
                    state_data["pumps"]["pump2_flow"][i],
                    state_data["setpoints"]["tank1"][i],
                    state_data["setpoints"]["tank2"][i],
                    state_data["setpoints"]["tank3"][i],
                    state_data["setpoints"]["tank4"][i],
                ]
                writer.writerow(row)
