#!/usr/bin/env python3
"""
SO100 Robot Control Demo - Interactive keyboard control
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_sim import task_suite
def get_key():
    """Get input with Enter key (macOS compatible)."""
    return input().strip().upper()

def control_robot():
    """Interactive robot control with real-time feedback."""
    print("SO100 Interactive Control Demo")
    
    # Create environment
    env = task_suite.create_task_env(
        task_name='SO100HandOverBanana',
        time_limit=300.0,  # 5 minutes
        cameras=('overhead_cam',),
        camera_resolution=(480, 640),
        image_observation_enabled=True,
    )
    
    timestep = env.reset()
    
    print("Controls:")
    print("  Q/A - Base rotation (left/right)")
    print("  W/S - Shoulder pitch (up/down)") 
    print("  E/D - Elbow (bend/extend)")
    print("  R/F - Wrist pitch (up/down)")
    print("  T/G - Wrist roll (rotate)")
    print("  Y/H - Gripper (open/close)")
    print("  SPACE - Stop all movement")
    print("  0 - Reset to home position")
    print("  X - Exit")
    print("\nType commands and press Enter to control the robot...")
    
    step_count = 0
    
    try:
        while True:
            print(f"\nStep {step_count} - Enter command: ", end='')
            key = get_key()
            
            # Initialize action (all zeros = no movement)
            action = np.zeros(6)
            movement_name = "No movement"
            
            # Map keys to actions
            if key == 'Q':
                action[0] = 0.3
                movement_name = "Base rotate LEFT"
            elif key == 'A':
                action[0] = -0.3
                movement_name = "Base rotate RIGHT"
            elif key == 'W':
                action[1] = 0.3
                movement_name = "Shoulder UP"
            elif key == 'S':
                action[1] = -0.3
                movement_name = "Shoulder DOWN"
            elif key == 'E':
                action[2] = 0.3
                movement_name = "Elbow BEND"
            elif key == 'D':
                action[2] = -0.3
                movement_name = "Elbow EXTEND"
            elif key == 'R':
                action[3] = 0.3
                movement_name = "Wrist pitch UP"
            elif key == 'F':
                action[3] = -0.3
                movement_name = "Wrist pitch DOWN"
            elif key == 'T':
                action[4] = 0.3
                movement_name = "Wrist roll CW"
            elif key == 'G':
                action[4] = -0.3
                movement_name = "Wrist roll CCW"
            elif key == 'Y':
                action[5] = 0.4
                movement_name = "Gripper OPEN"
            elif key == 'H':
                action[5] = -0.4
                movement_name = "Gripper CLOSE"
            elif key == '' or key == 'STOP':
                action = np.zeros(6)
                movement_name = "STOP"
            elif key == '0':
                timestep = env.reset()
                    step_count = 0
                continue
            elif key == 'X':
                    break
            else:
                print(f"Unknown key '{key}' - try again")
                continue
            
            # Execute action
            timestep = env.step(action)
            step_count += 1
            
            print(f"{movement_name}")
            print(f"   Action: {action}")
            print(f"   Reward: {timestep.reward:.3f}")
            
            # Save occasional frames
            if step_count % 10 == 0:
                observations = timestep.observation
                if 'overhead_cam' in observations:
                    from PIL import Image
                    img_data = observations['overhead_cam']
                    img = Image.fromarray(img_data.astype(np.uint8))
                    img.save(f'control_demo_step_{step_count}.png')
            
    except KeyboardInterrupt:
        except Exception as e:
        print(f"Error: {e}")

def simple_demo():
    """Simple pre-programmed demo sequence."""
    
    env = task_suite.create_task_env(
        task_name='SO100HandOverBanana',
        time_limit=60.0,
        cameras=('overhead_cam',),
        image_observation_enabled=True,
    )
    
    timestep = env.reset()
    
    # Demo sequence
    demo_moves = [
        ("Base rotation", [0.5, 0, 0, 0, 0, 0]),
        ("Shoulder movement", [0, 0.3, 0, 0, 0, 0]),
        ("Elbow bend", [0, 0, 0.4, 0, 0, 0]),
        ("Wrist positioning", [0, 0, 0, 0.3, 0.2, 0]),
        ("Gripper open", [0, 0, 0, 0, 0, 0.5]),
        ("Gripper close", [0, 0, 0, 0, 0, -0.3]),
        ("Return to center", [-0.2, -0.1, -0.2, -0.1, 0, 0])
    ]
    
    for i, (move_name, action) in enumerate(demo_moves):
        print(f"{i+1}. {move_name}: {action}")
        timestep = env.step(np.array(action))
        print(f"   Reward: {timestep.reward:.3f}")
        
        # Save frame
        observations = timestep.observation
        if 'overhead_cam' in observations:
            from PIL import Image
            img_data = observations['overhead_cam']
            img = Image.fromarray(img_data.astype(np.uint8))
            img.save(f'demo_move_{i+1}_{move_name.replace(" ", "_")}.png')
    

if __name__ == '__main__':
    print("Choose control mode:")
    print("1. Interactive keyboard control")
    print("2. Simple demo sequence")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        control_robot()
    elif choice == '2':
        simple_demo()
    else:
        print("Invalid choice. Running interactive control...")
        control_robot()