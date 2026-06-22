"""Quadruple tanks system model."""

import numpy as np
from typing import List, Dict, Tuple
from .tank import Tank, TankConfig


class QuadrupleTanksSystem:
    """
    Model of the quadruple tanks system.
    
    The system consists of 4 tanks arranged as follows:
    
         Pump 1        Pump 2
            ↓             ↓
          Tank 1        Tank 2
            ↓             ↓
          Tank 3        Tank 4
    
    The system has two pumps (inputs) and four tanks (outputs).
    Tanks 1 and 2 are upper tanks, tanks 3 and 4 are lower tanks.
    Water flows from upper tanks to lower tanks through orifices.
    """
    
    def __init__(self, tank_config: TankConfig = None):
        """
        Initialize the quadruple tanks system.
        
        Args:
            tank_config: TankConfig object (used for all tanks)
        """
        if tank_config is None:
            tank_config = TankConfig()
        
        # Create 4 tanks
        self.tanks: List[Tank] = [
            Tank(tank_id=1, config=tank_config),
            Tank(tank_id=2, config=tank_config),
            Tank(tank_id=3, config=tank_config),
            Tank(tank_id=4, config=tank_config),
        ]
        
        # Pump flow commands (now variable, controllable)
        self.pump1_flow_command = 250.0  # cm^3/s (commanded pump 1 flow)
        self.pump2_flow_command = 250.0  # cm^3/s (commanded pump 2 flow)
        self.pump1_flow_max = 300.0  # cm^3/s (max pump capacity)
        self.pump2_flow_max = 300.0  # cm^3/s (max pump capacity)
        
        # Fixed coupling split for upper tank outflows
        # Upper tank coupling fixed at 0.7 for consistent disturbance
        self.u14_fixed = 0.7
        self.u23_fixed = 0.7
        
        # Controlled lower tank drain valves (0: closed, 1: fully open)
        self.u3 = 0.5
        self.u4 = 0.5
        
        # Internal time accumulator for sinusoidal disturbance
        self._t = 0.0
        
        # History for logging
        self.time_history = []
        self.state_history = []
        
    def set_pump_flows(self, pump1_flow: float, pump2_flow: float) -> None:
        """
        Set pump flow commands for the system (new pump-based control).
        
        Args:
            pump1_flow: Pump 1 flow command [cm^3/s], clipped to [0, pump1_flow_max]
            pump2_flow: Pump 2 flow command [cm^3/s], clipped to [0, pump2_flow_max]
        """
        self.pump1_flow_command = np.clip(pump1_flow, 0.0, self.pump1_flow_max)
        self.pump2_flow_command = np.clip(pump2_flow, 0.0, self.pump2_flow_max)
    
    def set_valve_inputs(self, u13: float, u24: float, u3: float, u4: float) -> None:
        """
        Set valve openings for the system.
        
        Args:
            u13, u24: Coupling valve openings (kept for compatibility, ignored)
            u3: Drain valve opening for Tank 3 [0, 1]
            u4: Drain valve opening for Tank 4 [0, 1]
        """
        self.u3 = float(np.clip(u3, 0.0, 1.0))
        self.u4 = float(np.clip(u4, 0.0, 1.0))

    def set_drain_valves(self, u3: float, u4: float) -> None:
        """Set lower tank drain valve openings (0: closed, 1: fully open)."""
        self.set_valve_inputs(0.0, 0.0, u3, u4)
    
    def update(self, dt: float) -> Dict[int, float]:
        """
        Update system state for one time step using pump-based control.
        
        Args:
            dt: Time step [s]
            
        Returns:
            Dictionary with updated tank heights
        """
        # Accumulate simulation time
        self._t += dt
        
        # PUMP CONTROL: Use commanded pump flows (now variable)
        pump1_flow = self.pump1_flow_command
        pump2_flow = self.pump2_flow_command
        
        # Get baseline outflow rates from each upper tank
        outflow_1 = self.tanks[0].get_outflow_rate()
        outflow_2 = self.tanks[1].get_outflow_rate()
        
        # Calculate actual inflows to lower tanks based on FIXED coupling valves
        inflow_3 = self.u14_fixed * outflow_1 + self.u23_fixed * outflow_2
        inflow_4 = (1.0 - self.u14_fixed) * outflow_1 + (1.0 - self.u23_fixed) * outflow_2
        
        # Update tank states
        # Upper tanks: pump flows are the inputs, fixed coupling splits outflows
        self.tanks[0].update(pump1_flow, dt, outflow_multiplier=1.0)  # Tank 1: all pump flow in, all outflow split
        self.tanks[1].update(pump2_flow, dt, outflow_multiplier=1.0)  # Tank 2: all pump flow in, all outflow split
        
        # Lower tanks: receive coupling inflows and drain via controllable outlets
        self.tanks[2].update(inflow_3, dt, outflow_multiplier=self.u3)  # Tank 3 drain valve
        self.tanks[3].update(inflow_4, dt, outflow_multiplier=self.u4)  # Tank 4 drain valve
        
        return self.get_heights()
    
    def get_heights(self) -> Dict[int, float]:
        """
        Get current heights of all tanks.
        
        Returns:
            Dictionary with tank heights {tank_id: height}
        """
        return {tank.tank_id: tank.height for tank in self.tanks}
    
    def get_state(self) -> Dict:
        """
        Get complete system state.
        
        Returns:
            Dictionary with all system state variables
        """
        return {
            "heights": self.get_heights(),
            "pumps": {
                "pump1_flow_command": self.pump1_flow_command,
                "pump2_flow_command": self.pump2_flow_command,
                "pump1_flow_max": self.pump1_flow_max,
                "pump2_flow_max": self.pump2_flow_max,
            },
            "coupling_valves": {
                "u14_fixed": self.u14_fixed,
                "u23_fixed": self.u23_fixed,
            },
            "drain_valves": {
                "u3": self.u3,
                "u4": self.u4,
            },
            "tank_details": [tank.get_state() for tank in self.tanks],
        }
    
    def reset(self, initial_heights: Dict[int, float] = None) -> None:
        """
        Reset system to initial state.
        
        Args:
            initial_heights: Dictionary with initial heights for each tank
        """
        if initial_heights is None:
            initial_heights = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        for tank in self.tanks:
            height = initial_heights.get(tank.tank_id, 0.0)
            tank.reset(initial_height=height)
        
        self.u13 = 0.5
        self.u24 = 0.5
        self.u3 = 0.5
        self.u4 = 0.5
        self.time_history = []
        self.state_history = []
    
    def record_state(self, time: float) -> None:
        """
        Record current system state to history.
        
        Args:
            time: Current simulation time
        """
        self.time_history.append(time)
        self.state_history.append(self.get_state())
    
    def get_height(self, tank_id: int) -> float:
        """
        Get height of a specific tank.
        
        Args:
            tank_id: Tank identifier (1-4)
            
        Returns:
            Tank height [cm]
        """
        for tank in self.tanks:
            if tank.tank_id == tank_id:
                return tank.height
        raise ValueError(f"Tank {tank_id} not found")
    
    def get_relative_height(self, tank_id: int) -> float:
        """
        Get normalized height (0 to 1) of a specific tank.
        
        Args:
            tank_id: Tank identifier (1-4)
            
        Returns:
            Normalized height [0-1]
        """
        max_height = self.tanks[0].config.height_max
        return self.get_height(tank_id) / max_height
