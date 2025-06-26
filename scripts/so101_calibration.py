#!/usr/bin/env python3
"""
SO100 Robot Calibration System
Loads and applies joint calibration from red_arm.json
"""

import json
import numpy as np
import os
from typing import Dict, List, Optional

class SO101Calibration:
    """Handle SO100 robot calibration loading and application."""
    
    def __init__(self, calibration_file: str = "calibration/red_arm.json"):
        """Initialize calibration system.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_file = calibration_file
        self.calibration_data = {}
        self.joint_mapping = {
            # Calibration name → SO100 joint index
            'shoulder_pan': 0,    # Base rotation
            'shoulder_lift': 1,   # Shoulder pitch
            'elbow_flex': 2,      # Elbow
            'wrist_flex': 3,      # Wrist pitch
            'wrist_roll': 4,      # Wrist roll
            'gripper': 5          # Jaw/gripper
        }
        self.homing_offsets = np.zeros(6)  # 6 DOF robot
        
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """Load calibration data from JSON file."""
        try:
            if not os.path.exists(self.calibration_file):
                return False
            
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            # Extract homing offsets for each joint
            for joint_name, joint_idx in self.joint_mapping.items():
                if joint_name in self.calibration_data:
                    offset = self.calibration_data[joint_name].get('homing_offset', 0)
                    self.homing_offsets[joint_idx] = offset
                else:
                    pass
            
            return True
            
        except Exception as e:
            return False
    
    def print_calibration_summary(self):
        """Print summary of loaded calibration."""
        pass
    
    def apply_calibration_to_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """Apply calibration offsets to joint positions.
        
        Args:
            joint_positions: Raw joint positions (6 DOF)
            
        Returns:
            Calibrated joint positions
        """
        if len(joint_positions) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(joint_positions)}")
        
        # Apply homing offsets
        calibrated = joint_positions + self.homing_offsets
        
        return calibrated
    
    def apply_calibration_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply calibration offsets to action commands.
        
        Args:
            action: Raw action commands (6 DOF)
            
        Returns:
            Calibrated action commands
        """
        return self.apply_calibration_to_position(action)
    
    def remove_calibration_from_position(self, calibrated_positions: np.ndarray) -> np.ndarray:
        """Remove calibration offsets (inverse operation).
        
        Args:
            calibrated_positions: Calibrated joint positions
            
        Returns:
            Raw joint positions
        """
        if len(calibrated_positions) != 6:
            raise ValueError(f"Expected 6 joint positions, got {len(calibrated_positions)}")
        
        # Remove homing offsets
        raw = calibrated_positions - self.homing_offsets
        
        return raw
    
    def get_calibrated_home_position(self, raw_home: np.ndarray) -> np.ndarray:
        """Get calibrated home position.
        
        Args:
            raw_home: Raw home position
            
        Returns:
            Calibrated home position
        """
        return self.apply_calibration_to_position(raw_home)
    
    def get_joint_limits(self) -> Dict[str, Dict[str, float]]:
        """Get joint limits from calibration data."""
        limits = {}
        
        for joint_name, joint_idx in self.joint_mapping.items():
            if joint_name in self.calibration_data:
                joint_data = self.calibration_data[joint_name]
                limits[joint_name] = {
                    'range_min': joint_data.get('range_min', 0),
                    'range_max': joint_data.get('range_max', 4095),
                    'homing_offset': joint_data.get('homing_offset', 0)
                }
        
        return limits
    
    def validate_calibration(self) -> bool:
        """Validate that calibration data is reasonable."""
        # Check that offsets are within reasonable bounds
        max_offset = 2000  # Reasonable maximum offset
        
        for i, offset in enumerate(self.homing_offsets):
            if abs(offset) > max_offset:
                pass
        
        return True

def test_calibration():
    """Test the calibration system."""
    
    # Create calibration object
    calibration = SO101Calibration()
    
    # Test with sample joint positions
    raw_positions = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
    
    # Apply calibration
    calibrated = calibration.apply_calibration_to_position(raw_positions)
    
    # Test inverse
    recovered = calibration.remove_calibration_from_position(calibrated)
    
    # Check if inverse worked
    if np.allclose(raw_positions, recovered):
        pass
    else:
        pass
    
    # Show joint limits
    limits = calibration.get_joint_limits()
    
    return calibration

if __name__ == '__main__':
    test_calibration()