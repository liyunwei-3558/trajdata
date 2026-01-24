"""
SinD Interactive Bokeh Visualization - Simplified Version

This script demonstrates interactive visualization of SinD dataset using Bokeh.
Features:
- Single scene visualization with timestep slider
- Interactive animation through timesteps
- Agent trajectories with history and future
- Map visualization in the background

Usage:
    python test4_bokeh_simple.py

Controls:
- Slider: Navigate through timesteps
- Plot: Pan (click-drag) and zoom (scroll) are interactive
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import bokeh.plotting as plt
from bokeh.models import (
    ColumnDataSource,
    Div,
    Range1d,
    Slider,
)
from bokeh.layouts import column
from bokeh.plotting import figure, show
from bokeh.io import curdoc

from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajdata.caching.df_cache import DataFrameCache
from trajdata.maps.map_api import MapAPI
from trajdata.utils import vis_utils


class SinDBokehVisualizer:
    """Simplified Bokeh visualizer for SinD dataset."""

    def __init__(
        self,
        location: str = "xa",
        scene_idx: int = 0,
        history_sec: float = 2.0,
        future_sec: float = 3.0,
        map_radius: float = 60.0,
    ):
        """Initialize the visualizer.

        Args:
            location: SinD location (city) to visualize
            scene_idx: Index of the scene to visualize
            history_sec: History duration in seconds
            future_sec: Future duration in seconds
            map_radius: Map visualization radius in meters
        """
        self.location = location
        self.scene_idx = scene_idx
        self.history_sec = history_sec
        self.future_sec = future_sec
        self.map_radius = map_radius

        # Data path
        self.sind_data_dir = Path("/home/lyw/1TBSSD/Datasets/ClaudeWork/My_trajdata/datasets/SinD_dataset")
        self.cache_path = Path.home() / ".unified_data_cache"

        # Will be populated
        self.dataset = None
        self.scene = None
        self.scene_cache = None

    def load_data(self):
        """Load the SinD dataset and scene."""
        print(f"Loading SinD dataset for location: {self.location}")

        self.dataset = UnifiedDataset(
            desired_data=[f"sind-{self.location}"],
            data_dirs={"sind": str(self.sind_data_dir)},
            desired_dt=0.1,
            centric="agent",
            history_sec=(self.history_sec, self.history_sec),
            future_sec=(self.future_sec, self.future_sec),
            only_predict=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: 50.0),
            incl_robot_future=False,
            incl_raster_map=False,
            num_workers=0,
            verbose=False,
        )

        self.scene = self.dataset.get_scene(self.scene_idx)
        self.scene_cache = DataFrameCache(self.cache_path, self.scene)

        print(f"\nScene: {self.scene.name}")
        print(f"Location: {self.scene.location}")
        print(f"Length: {self.scene.length_timesteps} timesteps")
        print(f"dt: {self.scene.dt} seconds")
        print(f"Number of agents: {len(self.scene.agents)}")

        # Count vehicles
        vehicle_count = sum(
            1 for agent in self.scene.agents if agent.type == AgentType.VEHICLE
        )
        print(f"Number of vehicles: {vehicle_count}")

    def get_trajectory_at_timestep(self, timestep: int):
        """Get trajectory data for a specific timestep.

        Args:
            timestep: Timestep to visualize

        Returns:
            Dictionary with visualization data
        """
        scene = self.scene
        scene_cache = self.scene_cache

        # Get agents present at this timestep
        agents_at_ts = scene.agent_presence[timestep]
        vehicle_agents = [a for a in agents_at_ts if a.type == AgentType.VEHICLE]

        if not vehicle_agents:
            return None

        agent_ids = [agent.name for agent in vehicle_agents]

        # Calculate history and future timesteps
        hist_len = int(self.history_sec / scene.dt)
        fut_len = int(self.future_sec / scene.dt)

        # Prepare data
        all_hist_x = []
        all_hist_y = []
        all_hist_colors = []
        all_fut_x = []
        all_fut_y = []
        all_fut_colors = []
        curr_data = {
            "x": [],
            "y": [],
            "heading": [],
            "speed": [],
            "type_str": [],
        }

        for agent in vehicle_agents:
            # Get trajectory
            hist_x = []
            hist_y = []
            fut_x = []
            fut_y = []

            for ts in range(timestep - hist_len, timestep + fut_len + 1):
                if 0 <= ts < scene.length_timesteps:
                    try:
                        state = scene_cache.get_state(agent.name, ts)
                        pos_x = float(state.position[0])
                        pos_y = float(state.position[1])

                        if ts <= timestep:
                            hist_x.append(pos_x)
                            hist_y.append(pos_y)
                        else:
                            fut_x.append(pos_x)
                            fut_y.append(pos_y)
                    except KeyError:
                        break

            if hist_x and fut_x:
                all_hist_x.append(hist_x)
                all_hist_y.append(hist_y)
                all_hist_colors.append(vis_utils.get_agent_type_color(agent.type))
                all_fut_x.append(fut_x)
                all_fut_y.append(fut_y)
                all_fut_colors.append(vis_utils.get_agent_type_color(agent.type))

                # Current state
                curr_state = scene_cache.get_state(agent.name, timestep)
                curr_data["x"].append(float(curr_state.position[0]))
                curr_data["y"].append(float(curr_state.position[1]))
                curr_data["heading"].append(float(curr_state.heading[0]))
                curr_data["speed"].append(
                    float(np.linalg.norm(curr_state.velocity))
                )
                curr_data["type_str"].append("VEHICLE")

        return {
            "hist_x": all_hist_x,
            "hist_y": all_hist_y,
            "hist_colors": all_hist_colors,
            "fut_x": all_fut_x,
            "fut_y": all_fut_y,
            "fut_colors": all_fut_colors,
            "curr_data": curr_data,
        }

    def create_figure(self):
        """Create the Bokeh figure."""
        # Create figure
        TOOLTIPS = [
            ("Position", "(@x, @y) m"),
            ("Heading", "@heading rad"),
            ("Speed", "@speed m/s"),
            ("Type", "@type_str"),
        ]

        p = figure(
            width=1200,
            height=900,
            title=f"SinD {self.location} - {self.scene.name}",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
            tooltips=TOOLTIPS,
        )

        vis_utils.apply_default_settings(p)
        return p

    def update_plot(self, timestep: int):
        """Update the plot for a given timestep.

        Args:
            timestep: Timestep to visualize
        """
        # Get data
        data = self.get_trajectory_at_timestep(timestep)

        if data is None:
            print(f"No vehicles at timestep {timestep}")
            return

        # Clear existing renderers
        self.raw_figure.renderers = []
        self.raw_figure.title.text = f"SinD {self.location} - {self.scene.name} | Timestep: {timestep}/{self.scene.length_timesteps - 1} | Vehicles: {len(data['curr_data']['x'])}"

        # Get map
        try:
            map_api = MapAPI(self.cache_path)
            vec_map = map_api.get_map(f"sind:{self.location}")

            # Calculate extent from current positions
            x_coords = data["curr_data"]["x"]
            y_coords = data["curr_data"]["y"]
            x_center = sum(x_coords) / len(x_coords) if x_coords else 0
            y_center = sum(y_coords) / len(y_coords) if y_coords else 0

            x_min = x_center - self.map_radius
            x_max = x_center + self.map_radius
            y_min = y_center - self.map_radius
            y_max = y_center + self.map_radius

            # Draw map
            vis_utils.draw_map_elems(
                self.raw_figure,
                vec_map,
                np.eye(3),  # Identity transform for now
                (x_min, x_max, y_min, y_max),
            )
        except Exception as e:
            print(f"Map rendering skipped: {e}")

        # Draw history trajectories
        for i in range(len(data["hist_x"])):
            self.raw_figure.line(
                data["hist_x"][i],
                data["hist_y"][i],
                line_width=2,
                line_dash="dashed",
                color=data["hist_colors"][i],
                alpha=0.6,
            )

        # Draw future trajectories
        for i in range(len(data["fut_x"])):
            self.raw_figure.line(
                data["fut_x"][i],
                data["fut_y"][i],
                line_width=2,
                color=data["fut_colors"][i],
                alpha=0.8,
            )

        # Draw current positions
        curr_source = ColumnDataSource(data["curr_data"])
        self.raw_figure.scatter(
            x="x",
            y="y",
            size=150,
            fill_color="blue",
            line_color="black",
            fill_alpha=0.7,
            source=curr_source,
        )

        # Set range
        if data["curr_data"]["x"]:
            x_coords = data["curr_data"]["x"]
            y_coords = data["curr_data"]["y"]
            x_center = sum(x_coords) / len(x_coords)
            y_center = sum(y_coords) / len(y_coords)

            x_min = x_center - self.map_radius
            x_max = x_center + self.map_radius
            y_min = y_center - self.map_radius
            y_max = y_center + self.map_radius

            self.raw_figure.x_range = Range1d(x_min, x_max)
            self.raw_figure.y_range = Range1d(y_min, y_max)

    def setup_slider(self):
        """Setup the timestep slider."""
        max_timestep = self.scene.length_timesteps - 1

        slider = Slider(
            start=0,
            end=max_timestep,
            value=0,
            step=1,
            title="Timestep",
            width=1100,
            bar_color="lightblue",
        )

        # Status text
        status = Div(text=f"Timestep: 0 / {max_timestep}")

        # Callback
        def update_timestep(attr, old, new):
            timestep = int(new)
            status.text = f"Timestep: {timestep} / {max_timestep}"
            self.update_plot(timestep)

        slider.on_change("value", update_timestep)

        return slider, status

    def run(self):
        """Run the interactive visualization."""
        # Load data
        self.load_data()

        # Create figure
        self.raw_figure = self.create_figure()

        # Initial plot
        self.update_plot(0)

        # Setup slider
        slider, status = self.setup_slider()

        # Instructions
        instructions = Div(
            text="""
            <b>SinD Bokeh Interactive Visualization</b><br/>
            <b>Controls:</b><br/>
            • Use the slider to navigate through timesteps<br/>
            • Click and drag on the plot to pan<br/>
            • Scroll to zoom in/out<br/>
            <b>Legend:</b><br/>
            • Dashed lines: History trajectory<br/>
            • Solid lines: Future trajectory<br/>
            • Blue dots: Current vehicle positions
        """
        )

        # Create layout
        layout = column(
            instructions,
            slider,
            status,
            self.raw_figure,
        )

        # Show
        show(layout)


def main():
    """Main entry point."""
    # Configuration
    location = "xa"  # Xi'an
    scene_idx = 0  # First scene

    print("=" * 60)
    print("SinD Bokeh Interactive Visualization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Location: {location}")
    print(f"  Scene Index: {scene_idx}")

    # Create visualizer
    visualizer = SinDBokehVisualizer(
        location=location,
        scene_idx=scene_idx,
        history_sec=2.0,
        future_sec=3.0,
    )

    # Run
    visualizer.run()


if __name__ == "__main__":
    main()
