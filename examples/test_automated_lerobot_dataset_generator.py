import unittest
import os
import torch
import numpy as np
from automated_lerobot_dataset_generator import DatasetConfig, AutomatedLeRobotDatasetGenerator

class TestAutomatedLeRobotDatasetGenerator(unittest.TestCase):

    def setUp(self):
        self.output_dir = "./test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.config = DatasetConfig(
            num_episodes=1,  # Generate 1 main episode for testing
            max_episode_length=20, # Keep short for quick tests
            enable_failure_recovery=True,
            recovery_attempts=1
        )
        self.generator = AutomatedLeRobotDatasetGenerator(self.config)

    def tearDown(self):
        # Clean up generated files and directories
        for f in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, f))
        os.rmdir(self.output_dir)

    def test_dataset_generation_structure(self):
        print("\n--- Running test_dataset_generation_structure ---")
        dataset = self.generator.generate_dataset(save_path=os.path.join(self.output_dir, "test_dataset.pt"))

        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0, "Dataset should contain at least one episode.")

        # Check a sample episode
        sample_episode = dataset[0]
        self.assertIn("observations", sample_episode)
        self.assertIn("actions", sample_episode)
        self.assertIn("rewards", sample_episode)
        self.assertIn("dones", sample_episode)
        self.assertIn("episode_metadata", sample_episode)

        # Check observations structure
        obs = sample_episode["observations"]
        self.assertIn("image_overhead_cam", obs)
        self.assertIn("image_side_cam", obs)
        self.assertIn("image_wrist_cam", obs)
        self.assertIn("state", obs)

        # Verify lengths are consistent
        num_steps = len(sample_episode["actions"])
        self.assertEqual(len(obs["image_overhead_cam"]), num_steps + 1) # +1 for initial observation
        self.assertEqual(len(obs["state"]), num_steps + 1)
        self.assertEqual(len(sample_episode["rewards"]), num_steps)
        self.assertEqual(len(sample_episode["dones"]), num_steps + 1)

        # Check metadata
        metadata = sample_episode["episode_metadata"]
        self.assertIn("episode_id", metadata)
        self.assertIn("task_name", metadata)
        self.assertIn("success", metadata)
        self.assertIn("failure_reason", metadata)
        self.assertIn("is_recovery_episode", metadata)
        self.assertIn("initial_banana_pos", metadata)
        self.assertIn("initial_bowl_pos", metadata)
        self.assertIn("start_robot_pose", metadata)

        print("--- test_dataset_generation_structure passed ---")

    def test_recovery_episode_collection(self):
        print("\n--- Running test_recovery_episode_collection ---")
        # To force a failure, we can temporarily modify the success threshold
        original_grasp_threshold = self.generator.failure_detector.config.grasp_success_threshold
        self.generator.failure_detector.config.grasp_success_threshold = -1.0 # Force failure

        dataset = self.generator.generate_dataset(save_path=os.path.join(self.output_dir, "test_recovery_dataset.pt"))

        # Restore original threshold
        self.generator.failure_detector.config.grasp_success_threshold = original_grasp_threshold

        # Expect at least one main episode and one recovery episode
        self.assertGreaterEqual(len(dataset), 2, "Should have main and recovery episodes.")

        main_episode = None
        recovery_episode = None
        for ep in dataset:
            if not ep["episode_metadata"]["is_recovery_episode"]:
                main_episode = ep
            else:
                recovery_episode = ep
        
        self.assertIsNotNone(main_episode, "Main episode not found.")
        self.assertIsNotNone(recovery_episode, "Recovery episode not found.")

        self.assertFalse(main_episode["episode_metadata"]["success"], "Main episode should have failed.")
        self.assertTrue(recovery_episode["episode_metadata"]["is_recovery_episode"], "Episode should be marked as recovery.")
        self.assertEqual(recovery_episode["episode_metadata"]["parent_episode_id"], main_episode["episode_metadata"]["episode_id"], "Parent ID mismatch.")

        print("--- test_recovery_episode_collection passed ---")

if __name__ == '__main__':
    unittest.main()
