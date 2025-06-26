# SO100 Robot Documentation

Complete documentation for the SO100 real robot integration with Aloha simulation.

## 📚 Documentation Files

### 🤖 **[SO100_CONTROL_GUIDE.md](SO100_CONTROL_GUIDE.md)**
**Complete control and visualization guide**
- Robot control methods (keyboard, programming)
- 3D visualization options
- Camera access and observations
- Programming interface examples
- Troubleshooting guide

### 📷 **[how_to_camera.md](how_to_camera.md)**
**Camera configuration and management**
- Adding new cameras (fixed and mobile)
- Camera positioning and orientation
- Attachment to robot parts
- Testing and troubleshooting cameras
- Advanced camera features

### 🤖 **[lerobot_integration.md](lerobot_integration.md)**
**LeRobot dataset format integration**
- Convert SO100 observations to LeRobot format
- Data collection for training
- Camera mapping and specifications
- Training pipeline integration

## 🎯 Quick Start

### **1. See the Robot**
```bash
mjpython final_real_so100_viewer.py
```

### **2. Control the Robot**
```bash
python simple_so100_control.py
```

### **3. Programming Interface**
```python
from aloha_sim import task_suite

env = task_suite.create_task_env(
    task_name='SO100HandOverBanana',
    cameras=('overhead_cam', 'side_cam', 'wrist_cam'),
    image_observation_enabled=True,
)

timestep = env.reset()
action = [0.3, 0, 0, 0, 0, 0]  # Move base
timestep = env.step(action)
```

## 🤖 What You Have

### **Real SO-ARM100 Robot**
- ✅ **Orange/black colors** with detailed 3D meshes
- ✅ **6 degrees of freedom** (base, shoulder, elbow, wrist pitch/roll, gripper)
- ✅ **Task integration** (HandOver banana/pen/spoon)
- ✅ **Positioned correctly** on table with objects

### **Three Camera System**
- 📸 **overhead_cam** - Top-down strategic view
- 📸 **side_cam** - Fixed angled side view
- 📸 **wrist_cam** - **Mobile** first-person view (attached to robot wrist)

### **Control Methods**
- 🎮 **Interactive keyboard control** (`simple_so100_control.py`)
- 🖥️ **3D viewer** (`mjpython final_real_so100_viewer.py`)
- 💻 **Programming interface** (task_suite API)

## 🔧 File Structure

```
aloha_sim/
├── documentation/
│   ├── README.md                    # This file
│   ├── SO100_CONTROL_GUIDE.md      # Complete control guide  
│   └── how_to_camera.md            # Camera configuration guide
├── simple_so100_control.py         # Interactive control (recommended)
├── so100_control_demo.py          # Alternative control interface
├── final_real_so100_viewer.py     # 3D viewer with auto movement
├── wrist_camera_demo.py           # Dedicated wrist camera demo
└── aloha_sim/
    ├── tasks/
    │   ├── so100_hand_over.py      # SO100 HandOver tasks
    │   └── base/so100_task.py      # SO100 base task class
    └── assets/so100/
        ├── scene_pbr.xml           # Real SO100 scene with cameras
        └── assets/                 # Real SO-ARM100 mesh files (18 STL files)
```

## 🎯 Evolution Summary

**📦 Started with**: Gray geometric boxes  
**🎨 Improved to**: Better capsules and cylinders  
**🤖 Final result**: **REAL SO-ARM100 with actual meshes!**

## 📖 Additional Resources

- **MuJoCo Documentation**: Camera configuration and positioning
- **dm_control Documentation**: Task environment setup
- **robot_descriptions**: SO-ARM100 robot specifications

## 🔍 Need Help?

1. **Control issues**: See [SO100_CONTROL_GUIDE.md](SO100_CONTROL_GUIDE.md)
2. **Camera problems**: See [how_to_camera.md](how_to_camera.md)  
3. **Robot not visible**: Check positioning in scene_pbr.xml
4. **Keyboard control not working**: Use `simple_so100_control.py`

Your SO100 robot integration is complete! 🎉