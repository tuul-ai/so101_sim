#!/usr/bin/env python3
"""
SO100 LeRobot Dataset Wrapper
Converts SO100 observations to LeRobot dataset format for compatibility
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so101_sim import task_suite
from typing import Dict, Any, List, Optional

class SO101LeRobotWrapper:
    """Wrapper to convert SO100 observations to LeRobot format."""
    
    def __init__(
        self,
        task_name: str = 'SO100HandOverBanana',
        cameras: tuple = ('overhead_cam', 'side_cam', 'wrist_cam'),
        camera_resolution: tuple = (480, 640),
        time_limit: float = 30.0,
        device: str = 'cpu'
    ):
        """Initialize the LeRobot-compatible SO100 environment.
        
        Args:
            task_name: SO100 task name
            cameras: Camera names to include
            camera_resolution: (height, width) for cameras
            time_limit: Episode time limit
            device: torch device ('cpu' or 'cuda')
        """
        self.device = device
        self.cameras = cameras
        self.camera_resolution = camera_resolution
        
        # Use actual camera names (no mapping)
        self.use_actual_camera_names = True
        
        # Create SO100 environment
        self.env = task_suite.create_task_env(
            task_name=task_name,
            time_limit=time_limit,
            cameras=cameras,
            camera_resolution=camera_resolution,
            image_observation_enabled=True,
        )
        
        # Episode tracking
        self.episode_index = 0
        self.frame_index = 0
        self.start_time = 0.0
        
        print(f"   Cameras: {cameras}")
        print(f"   Resolution: {camera_resolution}")
        print(f"   Device: {device}")
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial observation in LeRobot format."""
        timestep = self.env.reset()
        
        # Reset episode tracking
        self.frame_index = 0
        self.start_time = 0.0
        
        return self._convert_to_lerobot_format(timestep, action=None)
    
    def step(self, action: np.ndarray) -> Dict[str, torch.Tensor]:
        """Step environment and return observation in LeRobot format."""
        timestep = self.env.step(action)
        self.frame_index += 1
        
        return self._convert_to_lerobot_format(timestep, action)
    
    def _convert_to_lerobot_format(
        self, 
        timestep, 
        action: Optional[np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert SO100 timestep to LeRobot dataset format."""
        lerobot_obs = {}
        
        # Convert images
        for camera_name in self.cameras:
            if camera_name in timestep.observation:
                # Get image data (H, W, C) uint8
                img_data = timestep.observation[camera_name]
                
                # Convert to tensor and normalize to [0, 1]
                img_tensor = torch.from_numpy(img_data).float() / 255.0
                
                # Transpose to (C, H, W) format
                img_tensor = img_tensor.permute(2, 0, 1)
                
                # Use actual camera name directly
                lerobot_obs[f'observation.images.{camera_name}'] = img_tensor.to(self.device)
        
        # Convert joint states
        if 'joints_pos' in timestep.observation:
            joint_positions = torch.from_numpy(timestep.observation['joints_pos']).float()
            lerobot_obs['observation.state'] = joint_positions.to(self.device)
        
        # Convert action
        if action is not None:
            action_tensor = torch.from_numpy(action).float()
            lerobot_obs['action'] = action_tensor.to(self.device)
        else:
            # Zero action for initial observation
            action_tensor = torch.zeros(6).float()
            lerobot_obs['action'] = action_tensor.to(self.device)
        
        # Add metadata
        lerobot_obs['timestamp'] = torch.tensor(self.frame_index * 0.1).float().to(self.device)
        lerobot_obs['frame_index'] = torch.tensor(self.frame_index).long().to(self.device)
        lerobot_obs['episode_index'] = torch.tensor(self.episode_index).long().to(self.device)
        lerobot_obs['index'] = torch.tensor(self.frame_index).long().to(self.device)
        lerobot_obs['task_index'] = torch.tensor(0).long().to(self.device)
        lerobot_obs['task'] = 'SO100 manipulation task'
        
        return lerobot_obs
    
    def collect_episode(
        self, 
        actions: List[np.ndarray], 
        save_path: Optional[str] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Collect a full episode and return in LeRobot format.
        
        Args:
            actions: List of actions to execute
            save_path: Optional path to save episode data
        
        Returns:
            List of observations in LeRobot format
        """
        episode_data = []
        
        # Reset environment
        obs = self.reset()
        episode_data.append(obs)
        
        
        # Execute actions
        for i, action in enumerate(actions):
            obs = self.step(action)
            episode_data.append(obs)
            
            if i % 10 == 0:
                print(f"   Step {i+1}/{len(actions)}")
        
        # Save if requested
        if save_path:
            torch.save(episode_data, save_path)
            
        self.episode_index += 1
        return episode_data
    
    def get_action_spec(self) -> Dict[str, Any]:
        """Get action specification."""
        return {
            'shape': (6,),
            'dtype': np.float32,
            'low': -1.0,
            'high': 1.0,
            'names': ['rotation', 'pitch', 'elbow', 'wrist_pitch', 'wrist_roll', 'jaw']
        }
    
    def get_observation_spec(self) -> Dict[str, Any]:
        """Get observation specification."""
        spec = {
            'images': {},
            'state': {
                'shape': (6,),
                'dtype': np.float32,
                'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
            }
        }
        
        # Add camera specs
        for camera_name in self.cameras:
            spec['images'][camera_name] = {
                'shape': (3, self.camera_resolution[0], self.camera_resolution[1]),
                'dtype': np.float32
            }
        
        return spec

def demo_lerobot_format():
    """Demonstrate the LeRobot format conversion."""
    print("=" * 50)
    
    # Create wrapper
    wrapper = SO101LeRobotWrapper(
        cameras=('overhead_cam', 'side_cam', 'wrist_cam'),
        device='cpu'
    )
    
    # Reset and get initial observation
    obs = wrapper.reset()
    
    for key in sorted(obs.keys()):
        if 'images' in key:
            print(f"   {key}: {obs[key].shape} (torch.Tensor)")
        elif isinstance(obs[key], torch.Tensor):
            print(f"   {key}: {obs[key].shape} (torch.Tensor)")
        else:
            print(f"   {key}: {type(obs[key])}")
    
    for key in obs.keys():
        if 'images' in key:
            tensor = obs[key]
            print(f"   {key}:")
            print(f"     Shape: {tensor.shape} (C, H, W)")
            print(f"     Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            print(f"     Dtype: {tensor.dtype}")
    
    print(f"   action: {obs['action'].shape} (torch.Tensor)")
    print(f"   Range: [{obs['action'].min():.4f}, {obs['action'].max():.4f}]")
    
    print(f"   observation.state: {obs['observation.state'].shape} (torch.Tensor)")
    print(f"   Joint positions: {obs['observation.state']}")
    
    # Test action step
    action = np.array([0.1, 0.2, 0.1, 0.0, 0.0, 0.0])
    obs_step = wrapper.step(action)
    
    print(f"   New action: {obs_step['action']}")
    print(f"   Frame index: {obs_step['frame_index']}")
    print(f"   Timestamp: {obs_step['timestamp']}")
    
    return wrapper

def create_lerobot_dataset():
    """Create a sample dataset in LeRobot format."""
    
    wrapper = SO101LeRobotWrapper()
    
    # Generate sample actions
    actions = []
    for i in range(20):
        # Simple sinusoidal movement
        t = i * 0.1
        action = np.array([
            0.3 * np.sin(t),      # Base rotation
            0.2 * np.sin(t * 1.5), # Shoulder
            0.4 * np.sin(t * 0.8), # Elbow
            0.1 * np.sin(t * 2.0), # Wrist pitch
            0.1 * np.sin(t * 1.2), # Wrist roll
            0.0                     # Gripper
        ])
        actions.append(action)
    
    # Collect episode
    episode_data = wrapper.collect_episode(actions, 'so100_lerobot_episode.pt')
    
    
    return episode_data

if __name__ == '__main__':
    # Run demo
    wrapper = demo_lerobot_format()
    
    # Create sample dataset
    dataset = create_lerobot_dataset()
    
    print(f"   from so100_lerobot_wrapper import SO101LeRobotWrapper")
    print(f"   wrapper = SO101LeRobotWrapper()")
    print(f"   obs = wrapper.reset()  # LeRobot format observation")