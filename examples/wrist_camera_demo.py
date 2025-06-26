#!/usr/bin/env python3
"""
Wrist Camera Demo - Shows first-person view from robot gripper
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_sim import task_suite

def wrist_camera_demo():
    print("Wrist Camera Demo")
    
    env = task_suite.create_task_env(
        task_name='SO100HandOverBanana',
        time_limit=120.0,
        cameras=('wrist_cam',),  # Only wrist camera
        camera_resolution=(480, 640),
        image_observation_enabled=True,
    )
    
    timestep = env.reset()
    print("Available commands:")
    print("  q/a - Base rotation")
    print("  w/s - Shoulder") 
    print("  e/d - Elbow")
    print("  r/f - Wrist pitch")
    print("  t/g - Wrist roll")
    print("  y/h - Gripper")
    print("  x - Exit")
    
    step = 0
    while True:
        cmd = input(f"\nStep {step} - Command: ").lower().strip()
        
        if cmd == 'x':
            break
            
        action = np.zeros(6)
        
        if cmd == 'q': action[0] = 0.3
        elif cmd == 'a': action[0] = -0.3
        elif cmd == 'w': action[1] = 0.3
        elif cmd == 's': action[1] = -0.3
        elif cmd == 'e': action[2] = 0.3
        elif cmd == 'd': action[2] = -0.3
        elif cmd == 'r': action[3] = 0.3
        elif cmd == 'f': action[3] = -0.3
        elif cmd == 't': action[4] = 0.3
        elif cmd == 'g': action[4] = -0.3
        elif cmd == 'y': action[5] = 0.3
        elif cmd == 'h': action[5] = -0.3
        else:
            print("Unknown command")
            continue
        
        timestep = env.step(action)
        step += 1
        
        # Save wrist camera view
        observations = timestep.observation
        if 'wrist_cam' in observations:
            from PIL import Image
            img_data = observations['wrist_cam']
            img = Image.fromarray(img_data.astype(np.uint8))
            filename = f'wrist_view_step_{step}.png'
            img.save(filename)
            
        print(f"Reward: {timestep.reward:.3f}")

if __name__ == '__main__':
    wrist_camera_demo()
