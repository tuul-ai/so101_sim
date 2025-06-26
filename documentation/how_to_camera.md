# Camera Configuration Guide for SO100 Robot

This guide explains how to add, modify, and configure cameras in your SO100 robot simulation.

## 📷 Current Camera Setup

Your SO100 system has **3 cameras**:

1. **overhead_cam** - Fixed top-down view
2. **side_cam** - Fixed angled side view  
3. **wrist_cam** - **Mobile** first-person view (attached to robot wrist)

## 🛠️ Camera Configuration Basics

### **Camera Definition in XML**

Cameras are defined in `/aloha_sim/assets/so100/scene_pbr.xml`:

```xml
<camera name="camera_name" pos="x y z" xyaxes="x1 y1 z1 x2 y2 z2" fovy="angle"/>
```

**Parameters:**
- `name`: Unique camera identifier
- `pos`: 3D position [x, y, z] in meters
- `xyaxes`: Orientation vectors (optional)
- `fovy`: Field of view angle in degrees

### **Current Camera Configurations**

```xml
<!-- Fixed cameras in worldbody -->
<camera name="overhead_cam" pos="0 0 1.5"/>
<camera name="side_cam" pos="0.8 -0.6 0.6" xyaxes="0.6 0.8 0 -0.32 0.24 0.92"/>
<camera name="worms_eye_cam" pos="0 0 0.1"/>

<!-- Mobile camera attached to robot wrist -->
<body name="Fixed_Jaw" pos="0 -0.0601 0" quat="0.707109 0 0.707105 0">
  <!-- ... other wrist components ... -->
  <camera name="wrist_cam" pos="0 -0.08 0" xyaxes="1 0 0 0 0 1" fovy="70"/>
</body>
```

## 🔧 How to Add a New Camera

### **Step 1: Add Camera to XML**

Edit `/aloha_sim/assets/so100/scene_pbr.xml`:

```xml
<!-- Example: Add a close-up table camera -->
<camera name="table_cam" pos="0.3 0.3 0.5" xyaxes="-0.707 -0.707 0 0.5 -0.5 0.707" fovy="45"/>
```

### **Step 2: Include in Task Environment**

Update your scripts to include the new camera:

```python
env = task_suite.create_task_env(
    task_name='SO100HandOverBanana',
    cameras=('overhead_cam', 'side_cam', 'wrist_cam', 'table_cam'),  # Add new camera
    camera_resolution=(480, 640),
    image_observation_enabled=True,
)
```

### **Step 3: Access Camera Data**

```python
timestep = env.reset()
table_view = timestep.observation['table_cam']  # Access new camera
```

## 📍 Camera Positioning Guide

### **Coordinate System**
- **X-axis**: Left/Right (positive = right)
- **Y-axis**: Forward/Back (positive = forward) 
- **Z-axis**: Up/Down (positive = up)
- **Origin (0,0,0)**: Floor level at table center

### **Robot and Scene Positions**
- **Robot base**: (0, 0, 0.42) - On table surface
- **Table surface**: Z = 0.42m
- **Banana**: (~0.2, 0.1, 0.42)
- **Bowl**: (~-0.2, 0.1, 0.42)

### **Camera Position Examples**

```xml
<!-- Top-down view -->
<camera name="overhead" pos="0 0 1.5"/>

<!-- Angled side view -->
<camera name="side" pos="0.8 -0.6 0.6" xyaxes="0.6 0.8 0 -0.32 0.24 0.92"/>

<!-- Close table view -->
<camera name="table_close" pos="0.5 0 0.8" xyaxes="-1 0 0 0 0 1"/>

<!-- Wide scene view -->
<camera name="wide" pos="1.5 1.5 1.0" xyaxes="-0.707 -0.707 0 0.408 -0.408 0.816"/>
```

### **Camera Orientation (xyaxes)**

The `xyaxes` parameter defines camera orientation:
- **Format**: `"x1 y1 z1 x2 y2 z2"`
- **x1,y1,z1**: Camera's right direction
- **x2,y2,z2**: Camera's up direction

**Common orientations:**
```xml
<!-- Looking forward -->
xyaxes="1 0 0 0 0 1"

<!-- Looking down -->  
xyaxes="1 0 0 0 1 0"

<!-- Looking at center from side -->
xyaxes="0.6 0.8 0 -0.32 0.24 0.92"
```

## 🔄 Mobile Cameras (Attached to Robot)

### **Attachment Points**

Cameras can be attached to any robot body:

```xml
<!-- Attach to robot base -->
<body name="Base" childclass="so_arm100" pos="0 0 0.42">
  <camera name="base_cam" pos="0 0 0.1" xyaxes="1 0 0 0 1 0"/>
</body>

<!-- Attach to upper arm -->
<body name="Upper_Arm">
  <camera name="arm_cam" pos="0 0.05 0" xyaxes="0 1 0 0 0 1"/>
</body>

<!-- Attach to wrist (current wrist_cam) -->
<body name="Fixed_Jaw">
  <camera name="wrist_cam" pos="0 -0.08 0" xyaxes="1 0 0 0 0 1" fovy="70"/>
</body>
```

### **Mobile Camera Behavior**
- **Moves with robot part** it's attached to
- **Inherits all transformations** (rotation, translation)
- **Perfect for first-person views** and manipulation monitoring

## 🎬 Camera Management in Code

### **Single Camera**
```python
env = task_suite.create_task_env(
    cameras=('overhead_cam',),  # Only overhead
)
```

### **Multiple Cameras**
```python
env = task_suite.create_task_env(
    cameras=('overhead_cam', 'side_cam', 'wrist_cam'),  # All three
)

# Access all camera data
observations = timestep.observation
overhead_img = observations['overhead_cam']
side_img = observations['side_cam'] 
wrist_img = observations['wrist_cam']
```

### **Save All Camera Views**
```python
def save_all_cameras(timestep, prefix="frame"):
    observations = timestep.observation
    cameras = ['overhead_cam', 'side_cam', 'wrist_cam']
    
    for cam_name in cameras:
        if cam_name in observations:
            from PIL import Image
            img_data = observations[cam_name]
            img = Image.fromarray(img_data.astype(np.uint8))
            img.save(f"{prefix}_{cam_name}.png")
```

## ⚙️ Camera Settings

### **Resolution**
```python
env = task_suite.create_task_env(
    camera_resolution=(480, 640),  # height, width
)
```

### **Field of View**
```xml
<camera name="wide_cam" fovy="90"/>    <!-- Wide angle -->
<camera name="normal_cam" fovy="45"/>  <!-- Normal -->
<camera name="zoom_cam" fovy="20"/>    <!-- Zoomed in -->
```

## 🎯 Common Camera Setups

### **1. Manipulation Setup**
```python
cameras=('overhead_cam', 'wrist_cam')  # Strategic + first-person
```

### **2. Full Monitoring**
```python
cameras=('overhead_cam', 'side_cam', 'wrist_cam')  # Complete coverage
```

### **3. Analysis Setup**
```python
cameras=('overhead_cam', 'side_cam', 'table_cam', 'wide_cam')  # Multiple angles
```

## 🔍 Testing New Cameras

### **Quick Test Script**
```python
#!/usr/bin/env python3
from aloha_sim import task_suite
from PIL import Image

def test_camera(camera_name):
    env = task_suite.create_task_env(
        task_name='SO100HandOverBanana',
        cameras=(camera_name,),
        image_observation_enabled=True,
    )
    
    timestep = env.reset()
    
    if camera_name in timestep.observation:
        img_data = timestep.observation[camera_name]
        img = Image.fromarray(img_data.astype(np.uint8))
        img.save(f"test_{camera_name}.png")
        print(f"✅ {camera_name} working - saved test_{camera_name}.png")
    else:
        print(f"❌ {camera_name} not found")

# Test your new camera
test_camera('your_new_camera_name')
```

## 🛠️ Troubleshooting

### **Camera Not Showing**
1. **Check name spelling** in XML and Python
2. **Verify camera position** isn't inside objects
3. **Test with simple position** like `pos="0 0 2"`

### **Wrong View Direction**
1. **Adjust xyaxes** parameter
2. **Use online MuJoCo camera calculators**
3. **Test incrementally** with small position changes

### **Camera Too Dark/Bright**
1. **Add more lights** in scene
2. **Adjust light positions** near camera view
3. **Check if camera is inside geometry**

### **Mobile Camera Not Moving**
1. **Verify attachment** to correct robot body
2. **Check body hierarchy** in XML
3. **Ensure camera is inside robot body tags**

## 📋 Camera Naming Conventions

- **overhead_cam**: Top-down views
- **side_cam**: Side profile views  
- **wrist_cam**: Attached to robot wrist
- **table_cam**: Close table views
- **wide_cam**: Wide scene views
- **gripper_cam**: Gripper-mounted cameras

## 🎥 Advanced Camera Features

### **Tracking Camera** (Always looks at target)
```xml
<camera name="tracking_cam" pos="1 1 1" mode="track" target="banana"/>
```

### **Camera with Specific Up Vector**
```xml
<camera name="level_cam" pos="0.5 0.5 0.8" xyaxes="0 1 0 0 0 1"/>
```

### **Multiple Mobile Cameras**
```xml
<!-- One on each arm segment -->
<body name="Upper_Arm">
  <camera name="upper_arm_cam" pos="0 0.05 0"/>
</body>
<body name="Lower_Arm">  
  <camera name="lower_arm_cam" pos="0 0.05 0"/>
</body>
<body name="Fixed_Jaw">
  <camera name="wrist_cam" pos="0 -0.08 0"/>
</body>
```

## 📖 Quick Reference

**Add Fixed Camera:**
1. Add `<camera>` tag in `<worldbody>`
2. Include name in `cameras=()` tuple
3. Access via `observation['camera_name']`

**Add Mobile Camera:**
1. Add `<camera>` tag inside robot `<body>`
2. Include name in `cameras=()` tuple  
3. Camera moves with robot part

**Change Camera Position:**
1. Edit `pos="x y z"` in XML
2. Restart environment to see changes

**Change Camera Direction:**  
1. Edit `xyaxes="..."` in XML
2. Use online calculators for complex orientations

Your cameras are now fully configurable! 📸🤖