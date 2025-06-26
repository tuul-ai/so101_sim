# SO100 Robot Calibration System

This guide explains the automatic robot calibration system that corrects for real-world hardware differences.

## 🎯 What is Robot Calibration?

**Robot calibration** corrects the difference between:
- **Theoretical positions** - Where the software thinks joints are
- **Actual positions** - Where the physical robot joints really are

This happens due to manufacturing tolerances, assembly variations, and wear over time.

## 🔧 Your Calibration File

Your robot uses calibration data from `calibration/red_arm.json`:

```json
{
    "shoulder_pan": {"homing_offset": 28, ...},     # Base rotation
    "shoulder_lift": {"homing_offset": 42, ...},    # Shoulder pitch  
    "elbow_flex": {"homing_offset": 18, ...},       # Elbow
    "wrist_flex": {"homing_offset": -21, ...},      # Wrist pitch
    "wrist_roll": {"homing_offset": 1009, ...},     # Wrist roll
    "gripper": {"homing_offset": -158, ...}         # Gripper/jaw
}
```

### **Key Parameters:**
- **`homing_offset`**: The main calibration value - joint position correction
- **`range_min`/`range_max`**: Joint movement limits
- **`id`**: Hardware joint ID
- **`drive_mode`**: Motor drive configuration

## 🚀 Automatic Integration

**Calibration is applied automatically** when you:

### **1. Create Any Task Environment**
```python
from aloha_sim import task_suite

env = task_suite.create_task_env('SO100HandOverBanana')
# 🔧 Calibration automatically loaded and applied!
```

### **2. Use LeRobot Wrapper**
```python
from so100_lerobot_wrapper import SO100LeRobotWrapper

wrapper = SO100LeRobotWrapper()
# 🔧 Calibration automatically loaded and applied!
```

### **3. Control the Robot**
```python
action = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
timestep = env.step(action)
# 🔧 Action automatically calibrated before applying to robot!
```

## 🔄 How Calibration Works

### **Initialization Process**
```
1. Task starts → Load calibration/red_arm.json
2. Apply offsets to home position
3. Move robot to calibrated home position
```

### **Action Process**
```
1. You send raw action: [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
2. System adds offsets:  [0.1+28, 0.2+42, 0.0+18, ...]
3. Robot receives:       [28.1, 42.2, 18.0, -21.0, 1009.0, -158.0]
```

### **Joint Mapping**
```python
# Calibration file → SO100 robot joints
'shoulder_pan'   → joint 0 (rotation/base)
'shoulder_lift'  → joint 1 (pitch/shoulder)
'elbow_flex'     → joint 2 (elbow)
'wrist_flex'     → joint 3 (wrist_pitch)
'wrist_roll'     → joint 4 (wrist_roll)
'gripper'        → joint 5 (jaw/gripper)
```

## 📊 Calibration Effects

### **Home Position Transformation**
```python
# Original home position
raw_home = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]

# Calibrated home position (automatically applied)
calibrated_home = [28.0, 40.43, 19.57, -19.43, 1007.43, -158.0]
```

### **Action Transformation Example**
```python
# Your command
action = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]

# What robot actually receives
calibrated = [28.1, 42.2, 18.0, -21.0, 1009.0, -158.0]

# Difference (the calibration offsets)
offsets = [28.0, 42.0, 18.0, -21.0, 1009.0, -158.0]
```

## 🛠️ Calibration File Management

### **Current Calibration File**
```
calibration/red_arm.json
```

### **Viewing Current Calibration**
```python
from so100_calibration import SO100Calibration

calibration = SO100Calibration()
# Prints calibration summary automatically
```

### **Updating Calibration**
1. **Edit calibration file**: Modify `homing_offset` values in `red_arm.json`
2. **Restart environment**: Create new task environment to load changes
3. **Test changes**: Use robot to verify calibration improves accuracy

### **Creating New Calibration File**
```json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": YOUR_VALUE,
        "range_min": 756,
        "range_max": 3226
    },
    // ... other joints
}
```

## 🔍 Testing Calibration

### **Quick Test**
```python
python so100_calibration.py
# Shows calibration loading and offset application
```

### **Integration Test**
```python
from aloha_sim import task_suite

env = task_suite.create_task_env('SO100HandOverBanana')
timestep = env.reset()
# Check console output for calibration messages
```

### **Manual Verification**
```python
# Compare positions before/after calibration
from so100_calibration import SO100Calibration

cal = SO100Calibration()
raw_pos = np.array([0, 0, 0, 0, 0, 0])
calibrated_pos = cal.apply_calibration_to_position(raw_pos)

print(f"Raw:        {raw_pos}")
print(f"Calibrated: {calibrated_pos}")
```

## 📋 Console Output

When calibration loads, you'll see:
```
🔧 Initializing SO100 calibration...
✅ Calibration loaded from: calibration/red_arm.json
📋 Joint Calibration Offsets:
   rotation    :     28.0
   pitch       :     42.0
   elbow       :     18.0
   wrist_pitch :    -21.0
   wrist_roll  :   1009.0
   jaw         :   -158.0
🏠 Calibrated home position: [  28.  40.43  19.57 -19.43 1007.43 -158. ]
```

## 🔧 Advanced Configuration

### **Different Calibration File**
```python
from so100_calibration import SO100Calibration

# Use custom calibration file
calibration = SO100Calibration("path/to/my_calibration.json")
```

### **Disable Calibration**
```python
# Create empty calibration file or set all offsets to 0
{
    "shoulder_pan": {"homing_offset": 0, ...},
    "shoulder_lift": {"homing_offset": 0, ...},
    // ... all zeros
}
```

### **Manual Calibration Application**
```python
from so100_calibration import SO100Calibration

cal = SO100Calibration()

# Apply to position
calibrated = cal.apply_calibration_to_position(raw_position)

# Remove calibration (inverse)
raw = cal.remove_calibration_from_position(calibrated_position)
```

## 🎯 Benefits

✅ **Automatic**: No manual intervention required  
✅ **Transparent**: Works with all existing code  
✅ **Accurate**: Corrects for real hardware differences  
✅ **Consistent**: Same calibration across all tasks  
✅ **Flexible**: Easy to update calibration values  

## 🔍 Troubleshooting

### **Calibration Not Loading**
- Check file path: `calibration/red_arm.json` exists
- Check JSON syntax: Use JSON validator
- Check permissions: File is readable

### **Robot Behaves Oddly**
- Check offset values: Very large offsets (>2000) may indicate problems
- Test with zero offsets: Temporarily set all offsets to 0
- Compare before/after: Test with and without calibration

### **Calibration File Missing**
```
Calibration file not found: calibration/red_arm.json
   Using zero offsets (no calibration)
```
System continues with no calibration applied.

## 📖 Summary

Your SO100 robot now **automatically applies calibration** from `red_arm.json`:

1. **Every task initialization** → Loads and applies calibration
2. **Every action command** → Gets calibrated before robot execution  
3. **No code changes needed** → Works transparently with existing code
4. **Real-world accuracy** → Corrects for hardware manufacturing differences

Your robot commands are now **automatically calibrated for real-world accuracy**! 🎯🤖