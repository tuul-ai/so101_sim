# SO100 Robot Control & Visualization Guide

## 🤖 Real SO-ARM100 Robot Control

Your SO100 robot has **6 degrees of freedom** with actual orange/black meshes:

### Joint Configuration
```python
action = [rotation, pitch, elbow, wrist_pitch, wrist_roll, jaw]
#         [   0   ,  1  ,   2  ,     3     ,     4     ,  5 ]
```

### Joint Details
1. **Rotation** (Base): Rotates the entire arm left/right
2. **Pitch** (Shoulder): Moves arm up/down  
3. **Elbow**: Bends the elbow joint
4. **Wrist Pitch**: Tilts the gripper up/down
5. **Wrist Roll**: Rotates the gripper 
6. **Jaw**: Opens/closes the gripper

## 🖥️ Visualization & Control Options

### **Option 1: Simple Interactive Control (Recommended)**
```bash
python simple_so100_control.py
```
- ✅ **Best for**: Learning and manual control
- ✅ **Compatible**: Works on all systems (macOS, Linux, Windows)
- ✅ **Input**: Type commands + Enter
- ✅ **Features**: Built-in demo, frame capture, help commands

### **Option 2: 3D Viewer with Auto Movement**
```bash
mjpython final_real_so100_viewer.py
```
- ✅ **Best for**: Visualizing the real robot in action
- ✅ **Features**: Automatic smooth robot movement
- ✅ **Visual**: Shows orange/black SO-ARM100 with detailed meshes
- ✅ **No input required**: Just watch the robot move

### **Option 3: Original Control Demo**
```bash
python so100_control_demo.py
```
- ✅ **Alternative**: If you prefer the original interface
- ✅ **Input**: Type commands + Enter (fixed for macOS)
- ✅ **Features**: Interactive and demo modes

## 🎮 Control Commands (All Options)

| Command | Action | Description |
|---------|--------|-------------|
| **q** | Base LEFT | Rotate base counter-clockwise |
| **a** | Base RIGHT | Rotate base clockwise |
| **w** | Shoulder UP | Lift shoulder/arm up |
| **s** | Shoulder DOWN | Lower shoulder/arm down |
| **e** | Elbow BEND | Bend elbow joint inward |
| **d** | Elbow EXTEND | Extend elbow joint outward |
| **r** | Wrist UP | Tilt wrist/gripper up |
| **f** | Wrist DOWN | Tilt wrist/gripper down |
| **t** | Wrist ROLL CW | Rotate gripper clockwise |
| **g** | Wrist ROLL CCW | Rotate gripper counter-clockwise |
| **y** | Gripper OPEN | Open gripper jaws |
| **h** | Gripper CLOSE | Close gripper jaws |
| **0** | RESET | Return to home position |
| **x** | EXIT | Close the program |
| **demo** | AUTO DEMO | Run automatic sequence *(simple_so100_control only)* |
| **help** | HELP | Show commands *(simple_so100_control only)* |

## 🎯 What You'll See

### **Real Robot Features:**
- 🤖 **Orange and black SO-ARM100** (not gray geometric shapes!)
- ✨ **Detailed 3D meshes** with proper joint segments
- 🔧 **6 realistic joints** matching real hardware
- 🏗️ **Positioned on brown table** with yellow banana and blue bowl

### **Available Tasks:**
1. **SO100HandOverBanana** - Move banana from table to bowl
2. **SO100HandOverPen** - Move pen to bowl  
3. **SO100HandOverSpoon** - Move spoon to bowl

## 💻 Programming Interface

### Basic Movement Example
```python
from aloha_sim import task_suite
import numpy as np

# Create environment with real robot
env = task_suite.create_task_env(
    task_name='SO100HandOverBanana',
    time_limit=30.0,
    cameras=('overhead_cam', 'side_cam'),
    image_observation_enabled=True,
)

# Reset to home position
timestep = env.reset()

# Example movements
base_rotation = [0.5, 0, 0, 0, 0, 0]      # Rotate base
shoulder_move = [0, 0.3, 0, 0, 0, 0]      # Move shoulder
elbow_bend = [0, 0, 0.5, 0, 0, 0]         # Bend elbow
wrist_move = [0, 0, 0, 0.3, 0.2, 0]       # Move wrist
gripper_open = [0, 0, 0, 0, 0, 0.5]       # Open gripper

# Execute action
timestep = env.step(np.array(base_rotation))
print(f"Reward: {timestep.reward}")

# Access camera observations
observations = timestep.observation
overhead_image = observations['overhead_cam']  # 480x640x3 RGB image
side_image = observations['side_cam']          # 480x640x3 RGB image
```

### Direct MuJoCo Control
```python
import mujoco

# Load the real SO100 scene
model = mujoco.MjModel.from_xml_path('aloha_sim/assets/so100/scene_pbr.xml')
data = mujoco.MjData(model)

# Direct joint control
data.ctrl[0] = 0.5  # Base rotation
data.ctrl[1] = 0.3  # Shoulder pitch
data.ctrl[2] = 0.4  # Elbow
data.ctrl[3] = 0.2  # Wrist pitch
data.ctrl[4] = 0.1  # Wrist roll
data.ctrl[5] = 0.3  # Gripper

# Step simulation
mujoco.mj_step(model, data)
```

## ⚙️ Settings & Ranges

### **Action Ranges:**
- **Position control**: -1.0 to 1.0 (normalized)
- **Safe testing range**: -0.5 to 0.5
- **Gripper**: -1.0 (closed) to 1.0 (open)

### **Camera Views:**
- **overhead_cam**: Top-down view (0, 0, 1.5m)
- **side_cam**: Angled side view (1.2, 0.8, 0.8m)
- **worms_eye_cam**: Bottom-up view (0, 0, 0.1m)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Robot moves too fast** | Reduce action values (use 0.1-0.3 instead of 0.5-1.0) |
| **Keyboard control not working** | Use `simple_so100_control.py` instead |
| **Robot under table** | Already fixed - robot now positioned at (0,0,0.42m) |
| **Gray geometric shapes** | Already fixed - now shows real orange/black meshes |
| **Joint limits exceeded** | Robot automatically respects joint limits |
| **Need to reset** | Type `0` in control or call `env.reset()` in code |

## 🚀 Quick Start

1. **See the robot**: `mjpython final_real_so100_viewer.py`
2. **Control the robot**: `python simple_so100_control.py`
3. **Try demo mode**: Choose option 2 in simple control or type `demo`

Your SO100 now has the **actual SO-ARM100 robot** with real meshes! 🎉