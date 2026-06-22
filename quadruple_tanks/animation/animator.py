"""Tank animation visualization using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional
from ..simulation import Simulator


class TankAnimator:
    """
    Real-time animation of quadruple tanks system.
    
    Visualizes water levels in all 4 tanks along with pump flows
    and setpoints.
    """
    
    def __init__(self, simulator: Simulator, figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize tank animator.
        
        Args:
            simulator: Simulator instance to animate
            figsize: Figure size (width, height)
        """
        self.simulator = simulator
        self.figsize = figsize
        
        # Create figure and axes
        self.fig = plt.figure(figsize=figsize)
        self.ax_tanks = self.fig.add_subplot(1, 2, 1)
        self.ax_plots = self.fig.add_subplot(1, 2, 2)
        
        # Tank parameters
        self.tank_width = 1.0
        self.tank_height = 2.0
        self.tank_positions = {
            1: (1.0, 3.0),    # Upper left
            2: (3.0, 3.0),    # Upper right
            3: (1.0, 0.5),    # Lower left
            4: (3.0, 0.5),    # Lower right
        }
        self.tank_colors = {
            1: '#FF6B6B',  # Red
            2: '#4ECDC4',  # Teal
            3: '#45B7D1',  # Blue
            4: '#FFA07A',  # Light salmon
        }
        self.tank_labels = {
            1: 'Tank 1\n(Upper Left)',
            2: 'Tank 2\n(Upper Right)',
            3: 'Tank 3\n(Lower Left)',
            4: 'Tank 4\n(Lower Right)',
        }
        
        # Storage for animation data
        self.time_data = []
        self.heights_data = {1: [], 2: [], 3: [], 4: []}
        self.valve_data = {"u13": [], "u24": [], "u3": [], "u4": []}
        
        # Scoring logic states - will be updated with per-tank setpoints
        self.tank_reached_time = {1: None, 2: None, 3: None, 4: None}
        self.setpoint1 = 75.0
        self.setpoint2 = 75.0
        self.setpoint3 = 75.0
        self.setpoint4 = 75.0
        self.max_duration = 300.0  # 5-minute simulation
        
        # Artist objects for updating
        self.water_rects = {}
        self.water_level_texts = {}
        self.tank_height_max = self.simulator.system.tanks[0].config.height_max
        
        # Setup tank visualization
        self._setup_tank_axes()
        
    def _setup_tank_axes(self) -> None:
        """Setup axes for tank visualization."""
        # Tank axes - expanded for better visualization
        self.ax_tanks.set_xlim(-1.5, 5.5)
        self.ax_tanks.set_ylim(-1.5, 6.0)
        self.ax_tanks.set_aspect('equal')
        self.ax_tanks.set_title('Quadruple Tanks System - Real-time Visualization', 
                                fontsize=14, fontweight='bold')
        self.ax_tanks.set_xlabel('Tank Position', fontsize=11, fontweight='bold')
        self.ax_tanks.set_ylabel('Height (cm)', fontsize=11, fontweight='bold')
        
        # Draw tank outlines and labels with improved styling
        for tank_id, (x, y) in self.tank_positions.items():
            # Tank outer frame (thicker border for 3D effect)
            rect_outer = patches.Rectangle(
                (x - self.tank_width/2 - 0.05, y - 0.05),
                self.tank_width + 0.1,
                self.tank_height + 0.1,
                linewidth=4,
                edgecolor='darkgray',
                facecolor='none',
                alpha=0.8
            )
            self.ax_tanks.add_patch(rect_outer)
            
            # Tank body (background)
            rect_body = patches.Rectangle(
                (x - self.tank_width/2, y),
                self.tank_width,
                self.tank_height,
                linewidth=2,
                edgecolor='black',
                facecolor='white',
                alpha=0.9
            )
            self.ax_tanks.add_patch(rect_body)
            
            # Water rectangle (will be updated)
            water_rect = patches.Rectangle(
                (x - self.tank_width/2, y),
                self.tank_width,
                0,  # Initial height 0
                linewidth=0,
                edgecolor='none',
                facecolor=self.tank_colors[tank_id],
                alpha=0.8
            )
            self.ax_tanks.add_patch(water_rect)
            self.water_rects[tank_id] = water_rect
            
            # Height scale markers on the side
            for h in range(0, int(self.tank_height) + 1, 1):
                marker_x = x - self.tank_width/2 - 0.15
                marker_y = y + h
                self.ax_tanks.plot([marker_x, marker_x + 0.1], [marker_y, marker_y], 
                                   'k-', linewidth=1.5, alpha=0.6)
                if h % 2 == 0:  # Label every 2 units
                    height_cm = (h / self.tank_height) * self.tank_height_max
                    self.ax_tanks.text(marker_x - 0.2, marker_y, f'{int(height_cm)}', 
                                     ha='right', va='center', fontsize=8, alpha=0.7)
            
            # Tank label above
            self.ax_tanks.text(
                x, y + self.tank_height + 0.5,
                self.tank_labels[tank_id],
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.tank_colors[tank_id], 
                         alpha=0.3, edgecolor=self.tank_colors[tank_id])
            )
            
            # Height text display (large, centered)
            text = self.ax_tanks.text(
                x, y + self.tank_height/2,
                '0.0\ncm',
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         alpha=0.8, edgecolor=self.tank_colors[tank_id], linewidth=2)
            )
            self.water_level_texts[tank_id] = text
            
            # Tank ID indicator
            self.ax_tanks.text(
                x + self.tank_width/2 + 0.15, y + self.tank_height - 0.2,
                f'T{tank_id}',
                ha='left', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=self.tank_colors[tank_id], 
                         alpha=0.6, edgecolor='black')
            )
        
        # Add interconnection indicators
        self._draw_connections()
        
        # Add max height reference line
        max_y = min([self.tank_positions[i][1] + self.tank_height for i in range(1, 5)])
        self.ax_tanks.axhline(y=max_y, color='red', linestyle='--', alpha=0.3, linewidth=1)
        self.ax_tanks.text(-1.3, max_y, 'Max', fontsize=9, color='red', 
                          va='center', fontweight='bold')
        
        self.ax_tanks.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        self.ax_tanks.set_facecolor('#f5f5f5')
        
        # Plot axes for trend monitoring
        self.ax_plots.set_title('System Trends', fontsize=12, fontweight='bold')
        self.ax_plots.set_xlabel('Time (s)')
        self.ax_plots.set_ylabel('Height (cm) / Flow (cm³/s)')
        self.ax_plots.grid(True, alpha=0.3)
        
        # Scoring text display (bottom-center, compact)
        self.score_text = self.ax_plots.text(
            0.5, 0.02, 'Score: —',
            transform=self.ax_plots.transAxes,
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.75, edgecolor='orange', linewidth=1.5)
        )
        
        self.lines_heights = {}
        self.lines_setpoints = {}
        colors = ['red', 'teal', 'blue', 'orange']
        
        for i, tank_id in enumerate([1, 2, 3, 4]):
            line, = self.ax_plots.plot([], [], color=colors[i], 
                                       label=f'Tank {tank_id}', linewidth=2)
            self.lines_heights[tank_id] = line
        
        # Setpoint lines for all tanks
        for i, tank_id in enumerate([1, 2, 3, 4]):
            line, = self.ax_plots.plot([], [], color=colors[i], 
                                       linestyle='--', alpha=0.5,
                                       label=f'SP {tank_id}', linewidth=1.5)
            self.lines_setpoints[tank_id] = line
    
    def _draw_connections(self) -> None:
        """Draw interconnection lines and flow indicators between tanks."""
        # Tank positions
        x1, y1 = self.tank_positions[1]
        x2, y2 = self.tank_positions[2]
        x3, y3 = self.tank_positions[3]
        x4, y4 = self.tank_positions[4]
        
        # Thicker, more visible connection lines
        # Tank 1 → Tank 3 (main connection)
        self.ax_tanks.annotate('', xy=(x3, y3 + self.tank_height), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', lw=2.5, color='darkred', 
                                            alpha=0.6, connectionstyle='arc3,rad=0.3'))
        
        # Tank 2 → Tank 4 (main connection)
        self.ax_tanks.annotate('', xy=(x4, y4 + self.tank_height), xytext=(x2, y2),
                              arrowprops=dict(arrowstyle='->', lw=2.5, color='darkblue', 
                                            alpha=0.6, connectionstyle='arc3,rad=-0.3'))
        
        # Cross connections (faint but visible)
        self.ax_tanks.annotate('', xy=(x4, y4 + self.tank_height*0.5), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', lw=1, color='gray', 
                                            alpha=0.4, connectionstyle='arc3,rad=0.5'))
        
        self.ax_tanks.annotate('', xy=(x3, y3 + self.tank_height*0.5), xytext=(x2, y2),
                              arrowprops=dict(arrowstyle='->', lw=1, color='gray', 
                                            alpha=0.4, connectionstyle='arc3,rad=-0.5'))
        
        # Pump indicators with better styling
        pump1_y = 5.2
        pump2_y = 5.2
        
        # Pump 1 with arrow
        self.ax_tanks.arrow(x1 - 0.3, pump1_y, 0.2, 0, head_width=0.2, 
                           head_length=0.15, fc='#00cc00', ec='darkgreen', linewidth=2, alpha=0.8)
        self.ax_tanks.text(x1 - 0.25, pump1_y + 0.35, 'u13 / u14(fixed)', fontsize=10, ha='center',
                          fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='lightgreen', alpha=0.7))
        
        # Pump 2 with arrow
        self.ax_tanks.arrow(x2 - 0.3, pump2_y, 0.2, 0, head_width=0.2, 
                           head_length=0.15, fc='#00cc00', ec='darkgreen', linewidth=2, alpha=0.8)
        self.ax_tanks.text(x2 - 0.25, pump2_y + 0.35, 'u24 / u23(fixed)', fontsize=10, ha='center',
                          fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='lightgreen', alpha=0.7))
                          
        # Tank 3 Drain (u3)
        self.ax_tanks.arrow(x3, y3, 0, -0.4, head_width=0.2, 
                           head_length=0.15, fc='#ff6600', ec='darkorange', linewidth=2, alpha=0.8)
        self.ax_tanks.text(x3, y3 - 0.7, 'u3 valve', fontsize=10, ha='center',
                          fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='peachpuff', alpha=0.7))
                          
        # Tank 4 Drain (u4)
        self.ax_tanks.arrow(x4, y4, 0, -0.4, head_width=0.2, 
                           head_length=0.15, fc='#ff6600', ec='darkorange', linewidth=2, alpha=0.8)
        self.ax_tanks.text(x4, y4 - 0.7, 'u4 valve', fontsize=10, ha='center',
                          fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                          facecolor='peachpuff', alpha=0.7))
        
        # Add system description
        description = "↓ Water flows down | → Cross-connects | ↓ Drain Outlets"
        self.ax_tanks.text(2.0, -1.2, description, fontsize=9, ha='center', 
                          style='italic', alpha=0.7)
    
    def animate(self, duration: float, 
                setpoint1: float = 75.0, 
                setpoint2: float = 75.0,
                setpoint3: float = 75.0,
                setpoint4: float = 75.0,
                interval: int = 100) -> FuncAnimation:
        """
        Create animation of the simulation.
        
        Args:
            duration: Simulation duration in seconds
            setpoint1: Setpoint for tank 1
            setpoint2: Setpoint for tank 2
            setpoint3: Setpoint for tank 3
            setpoint4: Setpoint for tank 4
            interval: Animation interval in milliseconds (lower = faster)
            
        Returns:
            FuncAnimation object
        """
        # Store per-tank setpoints for scoring
        self.setpoint1 = setpoint1
        self.setpoint2 = setpoint2
        self.setpoint3 = setpoint3
        self.setpoint4 = setpoint4
        
        self.simulator.set_setpoints(setpoint1, setpoint2, setpoint3, setpoint4)
        self.num_steps = int(duration / self.simulator.dt)
        self.current_step = 0
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._animate_frame,
            frames=self.num_steps,
            interval=interval,
            repeat=False,
            blit=False
        )
        
        return anim
    
    def _animate_frame(self, frame: int) -> List:
        """
        Update animation frame.
        
        Args:
            frame: Frame number
            
        Returns:
            List of artists to redraw
        """
        # Run simulator step
        self.simulator.step()
        
        # Collect data
        state = self.simulator.get_current_state()
        self.time_data.append(state['time'])
        
        for tank_id in range(1, 5):
            height = state['heights'][tank_id]
            self.heights_data[tank_id].append(height)
        
        # Store pump flow data instead of valve positions (pump control architecture)
        pump_flows = state.get('pumps', {'pump1': 250, 'pump2': 250})
        drain_valves = state.get('drain_valves', {'u3': 0.5, 'u4': 0.5})
        self.valve_data["u13"].append(pump_flows.get('pump1', 250) / 300.0)  # Normalize to [0,1]
        self.valve_data["u24"].append(pump_flows.get('pump2', 250) / 300.0)  # Normalize to [0,1]
        self.valve_data["u3"].append(drain_valves.get('u3', 0.5))
        self.valve_data["u4"].append(drain_valves.get('u4', 0.5))
        
        # Update tank water levels with enhanced visualization
        for tank_id in range(1, 5):
            height = state['heights'][tank_id]
            normalized_height = (height / self.tank_height_max) * self.tank_height
            
            rect = self.water_rects[tank_id]
            x, y = self.tank_positions[tank_id]
            rect.set_xy((x - self.tank_width/2, y))
            rect.set_height(max(0, min(normalized_height, self.tank_height)))
            
            # Update text with better formatting
            text = self.water_level_texts[tank_id]
            text.set_text(f'{height:.2f}\ncm')
            
            # Dynamic color intensity based on fill level
            fill_ratio = height / self.tank_height_max
            # Color becomes more saturated as tank fills
            color = self.tank_colors[tank_id]
            alpha = 0.3 + 0.7 * fill_ratio  # Alpha from 0.3 to 1.0
            rect.set_alpha(alpha)
            
            # Change border color to indicate status
            if fill_ratio > 0.95:
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#ffcccc', 
                                  alpha=0.9, edgecolor='red', linewidth=2))
            elif fill_ratio > 0.85:
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', 
                                  alpha=0.9, edgecolor='orange', linewidth=2))
            else:
                text.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  alpha=0.8, edgecolor=self.tank_colors[tank_id], linewidth=2))
                                  
            # Scoring tracking - now using per-tank setpoints
            if self.tank_reached_time[tank_id] is None:
                # Get this tank's setpoint
                setpoint = {
                    1: self.setpoint1,
                    2: self.setpoint2,
                    3: self.setpoint3,
                    4: self.setpoint4
                }[tank_id]
                tolerance = setpoint * 0.05  # ±5% of its setpoint
                
                if abs(height - setpoint) <= tolerance:
                    self.tank_reached_time[tank_id] = state['time']
        
        # Live Score: NEW FORMULA - sum individual tank contributions
        # Each tank that settles contributes (300 - t_settle) to total score
        setpoints_dict = {
            1: self.setpoint1,
            2: self.setpoint2,
            3: self.setpoint3,
            4: self.setpoint4
        }
        
        total_score = 0.0
        for tank_id in range(1, 5):
            sp = setpoints_dict[tank_id]
            tolerance = sp * 0.05  # ±5% of its setpoint
            if abs(state['heights'][tank_id] - sp) <= tolerance:
                if self.tank_reached_time[tank_id] is not None:
                    t_settle = self.tank_reached_time[tank_id]
                    if t_settle is not None:
                        tank_contribution = self.max_duration - t_settle
                        total_score += tank_contribution
        
        self.score_text.set_text(f'Score: {total_score:.0f}  |  Max possible: 1200')
        
        # Update trend plots
        self._update_trend_plots()
        
        # Update title with current time and pump status (pump control architecture)
        time_str = f'Time: {state["time"]:.1f}s'
        pump_flows = state.get('pumps', {'pump1': 250, 'pump2': 250})
        pump1_flow = pump_flows.get('pump1', 250)
        pump2_flow = pump_flows.get('pump2', 250)
        sim_status = f'Pump1: {pump1_flow:.0f} cm³/s | Pump2: {pump2_flow:.0f} cm³/s'
        self.ax_tanks.set_title(
            f'Quadruple Tanks System - {time_str} | {sim_status}',
            fontsize=14, fontweight='bold'
        )
        
        return list(self.water_rects.values()) + list(self.water_level_texts.values()) + [self.score_text]
    
    def _update_trend_plots(self) -> None:
        """Update trend plot lines."""
        time_array = np.array(self.time_data)
        
        # Update height lines with enhanced styling
        line_styles = ['-', '-', '--', '--']  # Solid for upper, dashed for lower
        line_widths = [2.5, 2.5, 2, 2]
        
        for i, tank_id in enumerate([1, 2, 3, 4]):
            heights = np.array(self.heights_data[tank_id])
            self.lines_heights[tank_id].set_data(time_array, heights)
            self.lines_heights[tank_id].set_linestyle(line_styles[i])
            self.lines_heights[tank_id].set_linewidth(line_widths[i])
        
        # Update setpoint lines with better visibility
        sp1_line = self.lines_setpoints[1]
        sp2_line = self.lines_setpoints[2]
        sp3_line = self.lines_setpoints[3]
        sp4_line = self.lines_setpoints[4]
        
        sp1_line.set_data(time_array, [self.simulator.setpoint1] * len(time_array))
        sp2_line.set_data(time_array, [self.simulator.setpoint2] * len(time_array))
        sp3_line.set_data(time_array, [self.simulator.setpoint3] * len(time_array))
        sp4_line.set_data(time_array, [self.simulator.setpoint4] * len(time_array))
        sp1_line.set_linewidth(2)
        sp2_line.set_linewidth(2)
        sp3_line.set_linewidth(2)
        sp4_line.set_linewidth(2)
        
        # Auto-scale axes with padding
        if len(time_array) > 0:
            max_time = max(time_array)
            self.ax_plots.set_xlim(-1, max_time * 1.05)
            self.ax_plots.set_ylim(-0.5, self.tank_height_max * 1.15)
            
            # Enhanced legend
            legend = self.ax_plots.legend(loc='upper left', fontsize=9, 
                                         framealpha=0.95, edgecolor='black', 
                                         title='Tank Levels & Setpoints', title_fontsize=10)
            legend.get_frame().set_linewidth(1.5)
            
            # Add grid improvements
            self.ax_plots.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
    
    def show(self) -> None:
        """Display the animation."""
        plt.tight_layout()
        plt.show()
    
    def save(self, filename: str, fps: int = 10) -> None:
        """
        Save animation to file.
        
        Args:
            filename: Output filename (e.g., 'animation.mp4')
            fps: Frames per second
        """
        try:
            from matplotlib.animation import PillowWriter
            
            writer = PillowWriter(fps=fps)
            # Note: This requires FFmpeg for MP4 or PIL for GIF
            self.fig.savefig(filename + '.png')
            print(f"✓ Static image saved to {filename}.png")
            
        except ImportError:
            print("⚠ Animation export requires additional packages")
            print("  For MP4: pip install ffmpeg-python")
            print("  For GIF: pip install Pillow")


def animate_simulation(simulator: Simulator, 
                      duration: float,
                      setpoint1: float = 75.0,
                      setpoint2: float = 75.0,
                      setpoint3: float = 75.0,
                      setpoint4: float = 75.0,
                      figsize: Tuple[int, int] = (14, 8),
                      interval: int = 100,
                      show_plot: bool = True) -> TankAnimator:
    """
    Quick function to animate a simulation.
    
    Args:
        simulator: Simulator instance
        duration: Simulation duration in seconds
        setpoint1: Setpoint for tank 1
        setpoint2: Setpoint for tank 2
        figsize: Figure size
        interval: Animation update interval in ms
        show_plot: Whether to display animation
        
    Returns:
        TankAnimator object
    """
    animator = TankAnimator(simulator, figsize=figsize)
    anim = animator.animate(duration, setpoint1, setpoint2, setpoint3, setpoint4, interval)
    
    if show_plot:
        animator.show()
    
    return animator
