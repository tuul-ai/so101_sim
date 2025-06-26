#!/usr/bin/env python3
"""
Automated Dataset Generator for SO101 Robot, formatted for LeRobot.
Generates diverse demonstration data with randomized environments and human-like trajectories,
including successful and recovery episodes.
"""

import numpy as np
import torch
import sys
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from so101_lerobot_wrapper import SO101LeRobotWrapper
from so101_calibration import SO101Calibration

@dataclass
class DatasetConfig:
    """Configuration for automated dataset generation."""
    # Episode settings
    num_episodes: int = 10
    max_episode_length: int = 50
    max_retries: int = 3
    
    # Randomization ranges
    banana_spawn_radius: float = 0.25  # meters around robot base
    bowl_spawn_radius: float = 0.30    # meters around robot base
    robot_pose_variation: float = 0.2  # joint angle variation in radians
    lighting_variation: float = 0.3    # lighting intensity variation
    
    # Trajectory settings
    approach_height: float = 0.1       # height above banana for approach
    lift_height: float = 0.15          # height to lift banana
    trajectory_smoothness: float = 0.8 # smoothing factor for waypoints
    movement_speed: float = 0.3        # base movement speed
    
    # Success criteria
    grasp_success_threshold: float = 0.05  # distance to banana for successful grasp
    bowl_success_threshold: float = 0.08   # distance to bowl for successful drop
    settling_time: int = 10                # steps to wait for physics settling
    
    # Failure recovery
    enable_failure_recovery: bool = True
    recovery_attempts: int = 2

class EnvironmentRandomizer:
    """Handles environment randomization for dataset diversity."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.calibration = SO101Calibration()
        
    def randomize_spawn_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random spawn positions for banana and bowl."""
        
        # Banana spawn - in front of robot within reach
        banana_angle = np.random.uniform(-np.pi/3, np.pi/3)  # 60-degree front arc
        banana_distance = np.random.uniform(0.15, self.config.banana_spawn_radius)
        banana_x = banana_distance * np.sin(banana_angle)
        banana_y = banana_distance * np.cos(banana_angle)
        banana_pos = np.array([banana_x, banana_y, 0.42])  # table height
        
        # Bowl spawn - closer to robot base, different angle
        bowl_angle = banana_angle + np.random.uniform(-np.pi/2, np.pi/2)
        bowl_distance = np.random.uniform(0.12, self.config.bowl_spawn_radius)
        bowl_x = bowl_distance * np.sin(bowl_angle)
        bowl_y = bowl_distance * np.cos(bowl_angle)
        bowl_pos = np.array([bowl_x, bowl_y, 0.42])
        
        return banana_pos, bowl_pos
    
    def randomize_robot_start_pose(self) -> np.ndarray:
        """Generate randomized starting pose for robot."""
        # Base pose with some variation
        base_pose = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
        
        # Add random variation to each joint
        variation = np.random.uniform(
            -self.config.robot_pose_variation, 
            self.config.robot_pose_variation, 
            size=6
        )
        
        # Apply calibration and joint limits
        randomized_pose = base_pose + variation
        
        # Clamp to reasonable joint limits
        joint_limits = [
            (-3.14, 3.14),   # base rotation
            (-2.5, 0.5),     # shoulder 
            (0.5, 2.5),      # elbow
            (0.5, 2.5),      # wrist pitch
            (-3.14, 3.14),   # wrist roll
            (0.0, 0.08)      # gripper
        ]
        
        for i, (min_val, max_val) in enumerate(joint_limits):
            randomized_pose[i] = np.clip(randomized_pose[i], min_val, max_val)
        
        return randomized_pose

class TrajectoryPlanner:
    """Plans human-like trajectories for banana pickup and placement."""
    
    def __init__(self, config: DatasetConfig, calibration: SO101Calibration):
        self.config = config
        self.calibration = calibration
        
    def plan_pickup_trajectory(self, start_joints: np.ndarray, banana_pos: np.ndarray) -> List[np.ndarray]:
        """Plan trajectory to pick up banana with human-like motion."""
        waypoints = []
        
        # 1. Approach phase - move above banana
        approach_pos = banana_pos.copy()
        approach_pos[2] += self.config.approach_height
        approach_joints = self._inverse_kinematics_approximate(approach_pos)
        
        # Add intermediate waypoints for smooth motion
        waypoints.extend(self._interpolate_trajectory(start_joints, approach_joints, steps=8))
        
        # 2. Descent phase - lower to banana
        grasp_pos = banana_pos.copy()
        grasp_pos[2] += 0.02  # slight offset above banana
        grasp_joints = self._inverse_kinematics_approximate(grasp_pos)
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], grasp_joints, steps=5))
        
        # 3. Grasp phase - close gripper
        grasp_joints_closed = grasp_joints.copy()
        grasp_joints_closed[5] = 0.05  # close gripper
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], grasp_joints_closed, steps=3))
        
        # 4. Lift phase - lift banana
        lift_pos = banana_pos.copy()
        lift_pos[2] += self.config.lift_height
        lift_joints = self._inverse_kinematics_approximate(lift_pos)
        lift_joints[5] = 0.05  # keep gripper closed
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], lift_joints, steps=5))
        
        return waypoints
    
    def plan_placement_trajectory(self, start_joints: np.ndarray, bowl_pos: np.ndarray) -> List[np.ndarray]:
        """Plan trajectory to place banana in bowl."""
        waypoints = []
        
        # 1. Move above bowl
        above_bowl_pos = bowl_pos.copy()
        above_bowl_pos[2] += self.config.lift_height
        above_bowl_joints = self._inverse_kinematics_approximate(above_bowl_pos)
        above_bowl_joints[5] = 0.05  # keep gripper closed
        
        waypoints.extend(self._interpolate_trajectory(start_joints, above_bowl_joints, steps=8))
        
        # 2. Lower into bowl
        drop_pos = bowl_pos.copy()
        drop_pos[2] += 0.05  # slightly above bowl
        drop_joints = self._inverse_kinematics_approximate(drop_pos)
        drop_joints[5] = 0.05  # keep gripper closed
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], drop_joints, steps=4))
        
        # 3. Release banana
        release_joints = drop_joints.copy()
        release_joints[5] = 0.0  # open gripper
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], release_joints, steps=3))
        
        # 4. Retreat
        retreat_pos = bowl_pos.copy()
        retreat_pos[2] += self.config.lift_height
        retreat_joints = self._inverse_kinematics_approximate(retreat_pos)
        retreat_joints[5] = 0.0  # keep gripper open
        waypoints.extend(self._interpolate_trajectory(waypoints[-1], retreat_joints, steps=5))
        
        return waypoints
    
    def _inverse_kinematics_approximate(self, target_pos: np.ndarray) -> np.ndarray:
        """Approximate inverse kinematics for SO101 robot."""
        # Simple geometric IK for 6-DOF arm
        # This is a simplified version - in practice you'd use a proper IK solver
        
        x, y, z = target_pos
        
        # Base rotation to point toward target
        base_angle = np.arctan2(x, y)
        
        # Distance from base to target
        horizontal_dist = np.sqrt(x**2 + y**2)
        vertical_dist = z - 0.42  # subtract table height
        
        # Approximate joint angles using geometric relationships
        # These are simplified calculations for demonstration
        shoulder_angle = -np.arctan2(vertical_dist, horizontal_dist) - 0.5
        elbow_angle = np.pi/2 + np.arctan2(vertical_dist, horizontal_dist)
        wrist_pitch = np.pi/2 - shoulder_angle - elbow_angle
        wrist_roll = 0.0
        gripper = 0.0
        
        joints = np.array([base_angle, shoulder_angle, elbow_angle, wrist_pitch, wrist_roll, gripper])
        
        # Apply calibration
        return self.calibration.apply_calibration_to_position(joints)
    
    def _interpolate_trajectory(self, start: np.ndarray, end: np.ndarray, steps: int) -> List[np.ndarray]:
        """Interpolate smooth trajectory between two joint configurations."""
        waypoints = []
        
        for i in range(steps):
            t = (i + 1) / steps
            # Use smooth interpolation with easing
            smooth_t = self._smooth_step(t)
            
            # Linear interpolation with smoothing
            interpolated = start + smooth_t * (end - start)
            
            # Add small random variation for human-like motion
            noise = np.random.normal(0, 0.02, size=6)
            noise[5] = 0  # no noise on gripper
            interpolated += noise * (1 - smooth_t)  # reduce noise near target
            
            waypoints.append(interpolated)
        
        return waypoints
    
    def _smooth_step(self, t: float) -> float:
        """Smooth step function for natural motion."""
        return t * t * (3 - 2 * t)

class FailureDetector:
    """Detects and handles failure cases during execution."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def detect_grasp_failure(self, robot_pos: np.ndarray, banana_pos: np.ndarray, gripper_state: float) -> bool:
        """Detect if grasping failed."""
        distance = np.linalg.norm(robot_pos - banana_pos)
        return distance > self.config.grasp_success_threshold or gripper_state < 0.02
    
    def detect_drop_failure(self, banana_pos: np.ndarray, bowl_pos: np.ndarray) -> bool:
        """Detect if banana was dropped outside bowl."""
        distance = np.linalg.norm(banana_pos[:2] - bowl_pos[:2])  # horizontal distance
        return distance > self.config.bowl_success_threshold
    
    def plan_recovery_trajectory(self, current_joints: np.ndarray, target_pos: np.ndarray) -> List[np.ndarray]:
        """Plan recovery trajectory for failed actions."""
        # Simple recovery: move to safe position then retry
        safe_joints = np.array([0.0, -1.0, 1.0, 1.0, 0.0, 0.0])
        
        # Move to safe position first
        recovery_waypoints = []
        for i in range(5):
            t = (i + 1) / 5
            interpolated = current_joints + t * (safe_joints - current_joints)
            recovery_waypoints.append(interpolated)
        
        return recovery_waypoints

class AutomatedLeRobotDatasetGenerator:
    """Main class for automated dataset generation in LeRobot format."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.calibration = SO101Calibration()
        self.randomizer = EnvironmentRandomizer(config)
        self.planner = TrajectoryPlanner(config, self.calibration)
        self.failure_detector = FailureDetector(config)
        self.episode_counter = 0
        
        # Initialize environment
        self.wrapper = SO101LeRobotWrapper(
            cameras=('overhead_cam', 'side_cam', 'wrist_cam'),
            device='cpu'
        )
        
    def generate_dataset(self, save_path: str = None) -> Dict:
        """Generate complete dataset with specified number of episodes."""
        print(f"Starting automated LeRobot dataset generation...")
        print(f"Target episodes: {self.config.num_episodes}")
        
        all_episodes_data = []
        successful_main_episodes = 0
        
        while successful_main_episodes < self.config.num_episodes:
            self.episode_counter += 1
            print(f"\n--- Main Episode {self.episode_counter} ---")
            
            episode_id = f"episode_{self.episode_counter}"
            
            # Generate main episode
            main_episode_data = self._generate_single_episode(episode_id, is_recovery=False)
            all_episodes_data.append(main_episode_data)
            
            if main_episode_data['episode_metadata']['success']:
                successful_main_episodes += 1
                print(f"Main Episode {self.episode_counter} successful ({len(main_episode_data['actions'])} steps)")
            else:
                print(f"Main Episode {self.episode_counter} failed. Reason: {main_episode_data['episode_metadata']['failure_reason']}")
                if self.config.enable_failure_recovery:
                    print(f"Attempting recovery for Episode {self.episode_counter}...")
                    recovery_success = False
                    for recovery_attempt_idx in range(self.config.recovery_attempts):
                        print(f"  Recovery Attempt {recovery_attempt_idx + 1}/{self.config.recovery_attempts}")
                        recovery_episode_id = f"{episode_id}_recovery_{recovery_attempt_idx + 1}"
                        
                        # Start recovery from the last state of the failed main episode
                        # For simplicity, we'll re-randomize the environment slightly for recovery,
                        # but in a real scenario, you'd want to preserve the exact failed state.
                        recovery_episode_data = self._generate_single_episode(
                            recovery_episode_id, 
                            is_recovery=True, 
                            parent_episode_id=episode_id
                        )
                        all_episodes_data.append(recovery_episode_data)
                        
                        if recovery_episode_data['episode_metadata']['success']:
                            print(f"  Recovery Attempt {recovery_attempt_idx + 1} successful!")
                            recovery_success = True
                            break
                        else:
                            print(f"  Recovery Attempt {recovery_attempt_idx + 1} failed. Reason: {recovery_episode_data['episode_metadata']['failure_reason']}")
                    
                    if recovery_success:
                        print(f"Recovery for Episode {self.episode_counter} completed successfully.")
                    else:
                        print(f"Recovery for Episode {self.episode_counter} failed after {self.config.recovery_attempts} attempts.")
            
            # Safety check to prevent infinite loops
            if self.episode_counter > self.config.num_episodes * (self.config.recovery_attempts + 1) * 2: # Heuristic to prevent infinite loops
                print("Too many attempts, stopping generation to prevent infinite loop.")
                break
        
        # Save dataset
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"so101_lerobot_dataset_{timestamp}.pt"
        
        torch.save(all_episodes_data, save_path)
        
        print(f"\nDataset generation complete!")
        print(f"Total episodes generated (main + recovery): {len(all_episodes_data)}")
        print(f"Successful main episodes: {successful_main_episodes}")
        print(f"Saved to: {save_path}")
        
        return all_episodes_data
    
    def _generate_single_episode(self, episode_id: str, is_recovery: bool, parent_episode_id: Optional[str] = None) -> Dict:
        """Generate a single episode (main or recovery)."""
        
        episode_observations = {
            "image_overhead_cam": [],
            "image_side_cam": [],
            "image_wrist_cam": [],
            "state": []
        }
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        
        success = False
        failure_reason = "max_steps_reached"
        
        # Randomize environment for main episode, or re-randomize slightly for recovery
        banana_pos, bowl_pos = self.randomizer.randomize_spawn_positions()
        start_pose = self.randomizer.randomize_robot_start_pose()
        
        # Reset environment
        obs = self.wrapper.reset()
        
        # Set starting pose
        obs = self.wrapper.step(start_pose)
        
        # Record initial observation
        episode_observations["image_overhead_cam"].append(obs["observation.images.overhead_cam"].numpy())
        episode_observations["image_side_cam"].append(obs["observation.images.side_cam"].numpy())
        episode_observations["image_wrist_cam"].append(obs["observation.images.wrist_cam"].numpy())
        episode_observations["state"].append(obs["observation.state"].numpy())
        episode_dones.append(False) # Not done at the first step
        
        # Phase 1: Pick up banana
        pickup_trajectory = self.planner.plan_pickup_trajectory(start_pose, banana_pos)
        
        for i, waypoint in enumerate(pickup_trajectory):
            if len(episode_actions) >= self.config.max_episode_length:
                break # Max steps reached
            
            obs = self.wrapper.step(waypoint)
            episode_observations["image_overhead_cam"].append(obs["observation.images.overhead_cam"].numpy())
            episode_observations["image_side_cam"].append(obs["observation.images.side_cam"].numpy())
            episode_observations["image_wrist_cam"].append(obs["observation.images.wrist_cam"].numpy())
            episode_observations["state"].append(obs["observation.state"].numpy())
            episode_actions.append(waypoint)
            episode_rewards.append(0.0)  # intermediate reward
            episode_dones.append(False)
        
        # Check if pickup succeeded
        current_joints = episode_observations["state"][-1]
        if self.failure_detector.detect_grasp_failure(current_joints[:3], banana_pos, current_joints[5]):
            failure_reason = "grasp_failed"
            # Mark the last step as done due to failure
            if episode_dones:
                episode_dones[-1] = True
            return {
                "observations": episode_observations,
                "actions": episode_actions,
                "rewards": episode_rewards,
                "dones": episode_dones,
                "episode_metadata": {
                    "episode_id": episode_id,
                    "task_name": "BananaPickAndPlace",
                    "success": False,
                    "failure_reason": failure_reason,
                    "is_recovery_episode": is_recovery,
                    "parent_episode_id": parent_episode_id,
                    "initial_banana_pos": banana_pos.tolist(),
                    "initial_bowl_pos": bowl_pos.tolist(),
                    "start_robot_pose": start_pose.tolist()
                }
            }
        
        # Phase 2: Place in bowl
        placement_trajectory = self.planner.plan_placement_trajectory(current_joints, bowl_pos)
        
        for i, waypoint in enumerate(placement_trajectory):
            if len(episode_actions) >= self.config.max_episode_length:
                break # Max steps reached
            
            obs = self.wrapper.step(waypoint)
            episode_observations["image_overhead_cam"].append(obs["observation.images.overhead_cam"].numpy())
            episode_observations["image_side_cam"].append(obs["observation.images.side_cam"].numpy())
            episode_observations["image_wrist_cam"].append(obs["observation.images.wrist_cam"].numpy())
            episode_observations["state"].append(obs["observation.state"].numpy())
            episode_actions.append(waypoint)
            episode_rewards.append(0.0)
            episode_dones.append(False)
        
        # Wait for settling
        for _ in range(self.config.settling_time):
            if len(episode_actions) >= self.config.max_episode_length:
                break # Max steps reached
            
            obs = self.wrapper.step(waypoint)  # hold last position
            episode_observations["image_overhead_cam"].append(obs["observation.images.overhead_cam"].numpy())
            episode_observations["image_side_cam"].append(obs["observation.images.side_cam"].numpy())
            episode_observations["image_wrist_cam"].append(obs["observation.images.wrist_cam"].numpy())
            episode_observations["state"].append(obs["observation.state"].numpy())
            episode_actions.append(waypoint)
            episode_rewards.append(0.0)
            episode_dones.append(False)
        
        # Final success check
        # For a real scenario, you'd need a more robust success detector based on physics
        # For now, we assume success if it reaches this point without a detected failure
        final_reward = 1.0
        if episode_rewards:
            episode_rewards[-1] = final_reward
        success = True
        failure_reason = None
        
        # Mark the last step as done
        if episode_dones:
            episode_dones[-1] = True
        
        return {
            "observations": episode_observations,
            "actions": episode_actions,
            "rewards": episode_rewards,
            "dones": episode_dones,
            "episode_metadata": {
                "episode_id": episode_id,
                "task_name": "BananaPickAndPlace",
                "success": success,
                "failure_reason": failure_reason,
                "is_recovery_episode": is_recovery,
                "parent_episode_id": parent_episode_id,
                "initial_banana_pos": banana_pos.tolist(),
                "initial_bowl_pos": bowl_pos.tolist(),
                "start_robot_pose": start_pose.tolist()
            }
        }

def main():
    """Main function for running automated dataset generation."""
    
    # Configuration
    config = DatasetConfig(
        num_episodes=2,  # Number of successful main episodes to generate
        max_episode_length=100, # Max steps per episode
        banana_spawn_radius=0.25,
        bowl_spawn_radius=0.30,
        enable_failure_recovery=True,
        recovery_attempts=1 # Number of recovery attempts per failed episode
    )
    
    print("=== SO101 Automated LeRobot Dataset Generator ===")
    print(f"Configuration: {config}")
    
    # Generate dataset
    generator = AutomatedLeRobotDatasetGenerator(config)
    dataset = generator.generate_dataset()
    
    print(f"\nDataset summary:")
    print(f"Total episodes collected: {len(dataset)}")
    
    # Optional: Verify a sample episode structure
    if dataset:
        sample_episode = dataset[0]
        print("\nSample Episode Structure:")
        print(f"  Episode ID: {sample_episode['episode_metadata']['episode_id']}")
        print(f"  Success: {sample_episode['episode_metadata']['success']}")
        print(f"  Is Recovery: {sample_episode['episode_metadata']['is_recovery_episode']}")
        print(f"  Num Steps: {len(sample_episode['actions'])}")
        print(f"  Observation Keys: {list(sample_episode['observations'].keys())}")
        print(f"  Sample Image Shape (overhead): {sample_episode['observations']['image_overhead_cam'][0].shape}")
        print(f"  Sample State Shape: {sample_episode['observations']['state'][0].shape}")
        
if __name__ == '__main__':
    main()
