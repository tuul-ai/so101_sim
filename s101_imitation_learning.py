import sys
import os
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from IPython import display # Import for in-notebook rendering

# Add paths for imports (assuming scripts directory is one level up)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from so101_lerobot_wrapper import SO101LeRobotWrapper
from scripts.image_utils import tensor_to_pil


class RealtimeRenderer:
    """
    A class for real-time rendering of robot observations.
    Can render in a standalone pop-up window or directly within a Jupyter notebook output.
    Press 'q' to close the window (standalone mode).
    """
    def __init__(self, camera_key: str = 'overhead_cam', title: str = "Robot View", interval: float = 0.03, in_notebook: bool = False):
        self.camera_key = camera_key
        self.title = title
        self.interval = interval
        self.in_notebook = in_notebook
        self.frame_count = 0
        
        if not self.in_notebook:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
            plt.ion()  # Turn on interactive mode
            plt.show(block=False)
            self.img_display = None

    def update(self, obs: dict):
        """
        Updates the display with the current observation.
        """
        if f'observation.images.{self.camera_key}' not in obs:
            print(f"Observation does not contain '{self.camera_key}' image.")
            return

        current_img = tensor_to_pil(obs[f'observation.images.{self.camera_key}'])

        if self.in_notebook:
            display.clear_output(wait=True) # Clear previous output
            plt.imshow(current_img)
            plt.title(f"{self.title} - Frame {self.frame_count}")
            plt.axis('off')
            display.display(plt.gcf()) # Display current figure
            time.sleep(self.interval) # Pause for pacing
        else:
            if self.img_display is None:
                self.img_display = self.ax.imshow(current_img)
            else:
                self.img_display.set_data(current_img)
            self.ax.set_title(f"{self.title} - Frame {self.frame_count}")
            self.ax.axis('off')  # Hide axes
            plt.draw()
            plt.pause(self.interval)

            # Check for 'q' key press to close the window
            if plt.waitforbuttonpress(timeout=0.001):
                if plt.gcf().canvas.manager.key_press_handler.key == 'q':
                    print("'q' pressed. Closing display window.")
                    self.close()
                    sys.exit() # Exit the script
        
        self.frame_count += 1

    def close(self):
        """
        Closes the rendering window or clears notebook output.
        """
        if self.in_notebook:
            display.clear_output(wait=True)
            print("Notebook rendering cleared.")
        else:
            plt.close(self.fig)
            print("Rendering window closed.")


# --- Simplified Simulation Loop for Demonstration (for standalone script) ---
def run_simulation_with_rendering():
    print("Initializing simulation for real-time rendering demonstration...")
    
    # Initialize the LeRobot wrapper
    lerobot = SO101LeRobotWrapper(
        task_name='SO100HandOverBanana', # Using a specific task for demonstration
        cameras=('overhead_cam', 'front_cam', 'wrist_cam'),
        camera_resolution=(480, 640),
        device='cpu'
    )
    
    obs = lerobot.reset()
    print("Environment reset. Starting real-time simulation loop...")

    # Initialize the renderer for standalone window
    renderer = RealtimeRenderer(camera_key='overhead_cam', in_notebook=False)

    # Simple mock action generation (replace with your actual policy/trajectory)
    num_steps = 100
    for step in range(num_steps):
        action = np.random.uniform(-0.01, 0.01, size=lerobot.get_action_spec()['shape'])
        action[-1] = 0.04 # Example gripper position

        obs = lerobot.step(action)
        renderer.update(obs)
        
        if (step + 1) % 20 == 0:
            print(f"Simulated {step + 1}/{num_steps} steps.")

    print("Real-time simulation finished.")
    renderer.close()

if __name__ == '__main__':
    run_simulation_with_rendering()