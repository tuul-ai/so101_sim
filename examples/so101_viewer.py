#!/usr/bin/env python3
"""
FINAL Real SO100 Viewer - Complete integration with real meshes
Run with: mjpython final_real_so100_viewer.py
"""

import mujoco
import mujoco.viewer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_sim import task_suite
import numpy as np

def main():
    print("SO100 3D Viewer")
    
    try:
        # Create task environment
        env = task_suite.create_task_env(
            task_name='SO100HandOverBanana',
            time_limit=120.0,
            cameras=(),
            image_observation_enabled=False,
        )
        
        timestep = env.reset()
        physics = env._physics
        
        
        with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
            step = 0
            while viewer.is_running():
                # Smooth, varied robot movement
                t = step * 0.01
                action = np.array([
                    1.0 * np.sin(t * 0.5),      # Base rotation
                    1.0 * np.sin(t * 0.3),      # Shoulder
                    1.0 * np.sin(t * 0.4),      # Elbow  
                    1.0 * np.sin(t * 0.6),      # Wrist pitch
                    1.0 * np.sin(t * 0.8),      # Wrist roll
                    0.5 * np.sin(t * 1.0)       # Gripper
                ])
                
                timestep = env.step(action)
                viewer.sync()
                step += 1
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()