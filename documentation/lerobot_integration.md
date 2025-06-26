# LeRobot Dataset Integration for SO100

This guide explains how to use SO100 with LeRobot dataset format for training robot learning models.

## 🎯 Overview

The SO100 LeRobot wrapper converts standard SO100 observations to LeRobot-compatible format:

**Standard SO100 Format:**
```python
{
    'overhead_cam': array(..., dtype=uint8),      # (H, W, C) uint8 [0-255]
    'side_cam': array(..., dtype=uint8),
    'wrist_cam': array(..., dtype=uint8),
    'joints_pos': array([...]),
    'physics_state': array([...]),
    ...
}
```

**LeRobot Format:**
```python
{
    'observation.images.overhead_cam': tensor(...), # (C, H, W) float32 [0-1]
    'observation.images.side_cam': tensor(...), 
    'observation.images.wrist_cam': tensor(...),
    'observation.state': tensor([...]),            # Joint positions
    'action': tensor([...]),                       # Robot actions
    'timestamp': tensor(...),
    'frame_index': tensor(...),
    'episode_index': tensor(...),
    'task': 'SO100 manipulation task'
}
```

## 🛠️ Usage

### **Basic Setup**

```python
from so100_lerobot_wrapper import SO100LeRobotWrapper

# Create wrapper
wrapper = SO100LeRobotWrapper(
    task_name='SO100HandOverBanana',
    cameras=('overhead_cam', 'side_cam', 'wrist_cam'),
    camera_resolution=(480, 640),
    device='cpu'  # or 'cuda'
)

# Reset environment
obs = wrapper.reset()

# Execute action
action = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
obs = wrapper.step(action)

# obs is now in LeRobot format!
```

### **Data Collection**

```python
# Interactive demonstration collection
python collect_lerobot_data.py
```

Choose option 1 for interactive control:
- Control robot with keyboard
- Each action is recorded in LeRobot format
- Save demonstrations for training

### **Automated Episode Collection**

```python
# Collect multiple episodes automatically
wrapper = SO100LeRobotWrapper()

actions = [
    np.array([0.2, 0.3, 0.1, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.2, 0.1, 0.0, 0.3]),
    # ... more actions
]

episode_data = wrapper.collect_episode(actions, 'episode_001.pt')
```

## 📊 Data Format Details

### **Image Observations**

```python
# Camera names are used directly (no mapping)
overhead_cam → observation.images.overhead_cam
side_cam     → observation.images.side_cam  
wrist_cam    → observation.images.wrist_cam
table_cam    → observation.images.table_cam
```

**Properties:**
- **Format**: `(C, H, W)` - Channels first
- **Type**: `torch.float32`
- **Range**: `[0.0, 1.0]` - Normalized
- **Channels**: `RGB` (3 channels)

### **State Observations**

```python
obs['observation.state']  # torch.Tensor shape (6,)
# [rotation, pitch, elbow, wrist_pitch, wrist_roll, jaw]
```

### **Actions**

```python
obs['action']  # torch.Tensor shape (6,)
# [rotation_cmd, pitch_cmd, elbow_cmd, wrist_pitch_cmd, wrist_roll_cmd, jaw_cmd]
# Range: [-1.0, 1.0] (normalized)
```

### **Metadata**

```python
obs['timestamp']      # float32 - Time in episode
obs['frame_index']    # int64 - Frame number in episode  
obs['episode_index']  # int64 - Episode number
obs['task']          # str - Task description
```

## 🎬 Data Collection Workflows

### **1. Interactive Demonstration**

```bash
python collect_lerobot_data.py
# Choose option 1
# Control robot with: q/a w/s e/d r/f t/g y/h
# Type 'done' to finish episode
# Data saved automatically
```

### **2. Scripted Collection**

```python
from collect_lerobot_data import collect_multiple_episodes

episodes, filename = collect_multiple_episodes(num_episodes=10)
# Automatically generates varied demonstration sequences
```

### **3. Custom Collection**

```python
wrapper = SO100LeRobotWrapper()

# Your custom action sequence
actions = generate_custom_actions()

# Collect episode
episode_data = wrapper.collect_episode(actions)

# Save
torch.save(episode_data, 'custom_episode.pt')
```

## 💾 Dataset Structure

### **Single Episode File**
```python
episode_data = [
    {obs_0},  # Initial observation
    {obs_1},  # After action 0
    {obs_2},  # After action 1
    ...
]
```

### **Multi-Episode Dataset**
```python
dataset = {
    'episodes': [
        [obs_0, obs_1, ...],  # Episode 0
        [obs_0, obs_1, ...],  # Episode 1
        ...
    ],
    'metadata': {
        'num_episodes': 10,
        'total_steps': 250,
        'cameras': ('overhead_cam', 'side_cam', 'wrist_cam'),
        'resolution': (480, 640),
        'action_spec': {...},
        'observation_spec': {...}
    }
}
```

## 🔧 Camera Configuration

### **Camera Name Usage**
```python
# Camera names are used directly in observations
cameras = ('overhead_cam', 'side_cam', 'wrist_cam', 'table_cam')

# Results in observation keys:
# - observation.images.overhead_cam  # Top-down view
# - observation.images.side_cam      # Side profile view
# - observation.images.wrist_cam     # Gripper perspective  
# - observation.images.table_cam     # Close table view
```

### **Custom Camera Setup**
```python
wrapper = SO100LeRobotWrapper(
    cameras=('overhead_cam', 'side_cam', 'wrist_cam', 'table_cam'),
    camera_resolution=(640, 480),  # Higher resolution
)

# Results in observations:
# - observation.images.overhead_cam
# - observation.images.side_cam  
# - observation.images.wrist_cam
# - observation.images.table_cam
```

## 🎯 Training Integration

### **Loading Dataset for Training**

```python
import torch

# Load single episode
episode = torch.load('so100_demo_episode_20241225_143022.pt')

# Load multi-episode dataset
dataset = torch.load('so100_dataset_20241225_143500.pt')
episodes = dataset['episodes']
metadata = dataset['metadata']

# Iterate through data
for episode in episodes:
    for obs in episode:
        images = {
            'overhead_cam': obs['observation.images.overhead_cam'],
            'side_cam': obs['observation.images.side_cam'], 
            'wrist_cam': obs['observation.images.wrist_cam']
        }
        state = obs['observation.state']
        action = obs['action']
        
        # Use for training...
```

### **Compatible with LeRobot Training**

The format is directly compatible with LeRobot training pipelines:

```python
# Example training loop (pseudo-code)
for batch in dataloader:
    images = batch['observation.images']  # Dict of camera views
    state = batch['observation.state']    # Robot state
    actions = batch['action']             # Target actions
    
    # Forward pass
    predicted_actions = model(images, state)
    
    # Loss computation
    loss = criterion(predicted_actions, actions)
    loss.backward()
```

## 📋 Specifications

### **Action Space**
```python
{
    'shape': (6,),
    'dtype': np.float32,
    'low': -1.0,
    'high': 1.0,
    'names': ['rotation', 'pitch', 'elbow', 'wrist_pitch', 'wrist_roll', 'jaw']
}
```

### **Observation Space**
```python
{
    'images': {
        'overhead_cam': {'shape': (3, 480, 640), 'dtype': np.float32},
        'side_cam': {'shape': (3, 480, 640), 'dtype': np.float32},
        'wrist_cam': {'shape': (3, 480, 640), 'dtype': np.float32}
    },
    'state': {
        'shape': (6,),
        'dtype': np.float32,
        'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    }
}
```

## 🔍 Debugging and Inspection

### **Inspect Dataset**
```python
python collect_lerobot_data.py
# Choose option 3
# Enter filename to inspect
```

### **Quick Visualization**
```python
from PIL import Image
import torch

# Load episode
episode = torch.load('episode.pt')
obs = episode[10]  # Frame 10

# Convert back to PIL for viewing
for camera in ['overhead_cam', 'side_cam', 'wrist_cam']:
    tensor = obs[f'observation.images.{camera}']
    
    # Convert (C,H,W) back to (H,W,C) and scale to 0-255
    img_array = (tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f'debug_{camera}.png')
```

## 🚀 Quick Start Example

```python
#!/usr/bin/env python3
from so100_lerobot_wrapper import SO100LeRobotWrapper
import numpy as np
import torch

# Create wrapper
wrapper = SO100LeRobotWrapper()

# Collect demonstration
actions = [
    np.array([0.2, 0.3, 0.1, 0.0, 0.0, 0.3]),  # Reach and open gripper
    np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),  # Lower down
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.3]), # Close gripper
    np.array([-0.2, -0.1, -0.1, 0.0, 0.0, 0.0]) # Move away
]

episode = wrapper.collect_episode(actions, 'my_demo.pt')

print(f"✅ Collected {len(episode)} observations")
print(f"💾 Saved LeRobot-format dataset: my_demo.pt")
```

Your SO100 data is now ready for LeRobot training! 🤖📊