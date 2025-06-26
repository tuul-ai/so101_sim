# SO101_Sim

<a href="#"><img alt="A banner with the title: SO101_Sim" src="media/banner.png" width="100%"></a>

SO101_Sim is a python library that defines the sim environment for the SO100/SO101 robotic arm.
It includes a collection of tasks for robot learning and evaluation.

## Installation

Install with pip:

```bash
# create a virtual environment and pip install
pip install -e .
```

**Note:** The `safari-sdk` dependency is commented out in `requirements.txt` for macOS compatibility.

**OR** run directly with uv:

```bash
pip install uv
uv run <script>.py
```

Tell mujoco which backend to use, otherwise the simulation will be very slow

```bash
export MUJOCO_GL='egl'
```

## Reinforcement Learning

This repository includes a collection of tasks for robot learning and evaluation. You can use the provided `so101_rl.ipynb` notebook to get started with training your own reinforcement learning agents.

### LeRobot Integration

The `so101_il_lerobot_format.ipynb` notebook demonstrates how to use the `SO101LeRobotWrapper` to interface with the environment in a format that is compatible with the [LeRobot](https://github.com/huggingface/lerobot) library.

## Viewer

Interact with the scene without a policy:

```bash
python so101_sim/viewer.py
```

## Tests

```bash
# individual tests
python so101_sim/tasks/test/hand_over_test.py
...

# all tests
python -m unittest discover so101_sim/tasks/test '*_test.py'
```


## Tips

- If the environment stepping is very slow, check that you are using the right
backend, e.g. `MUJOCO_GL='egl'`
- Tasks with deformable objects like `DesktopWrapHeadphone` and
`TowelFoldInHalf` are slow to simulate and interact directly with `viewer.py`.

## Automated LeRobot Dataset Generation

This script allows you to automatically generate demonstration data for imitation learning in the LeRobot format. It simulates "banana pick and place" tasks, including both successful and recovery episodes.

### Usage

1.  **Ensure Dependencies:** Make sure you have `uv` installed and `MUJOCO_GL` environment variable set as described in the [Installation](#installation) section.

2.  **Run the Generator:**
    To generate a dataset, execute the script from the project root:

    ```bash
    python examples/automated_lerobot_dataset_generator.py
    ```

3.  **Configuration:**
    You can customize the data generation process by modifying the `DatasetConfig` class within `examples/automated_lerobot_dataset_generator.py`. Key parameters include:
    *   `num_episodes`: Number of successful main episodes to generate.
    *   `max_episode_length`: Maximum number of steps per episode.
    *   `enable_failure_recovery`: Set to `True` to enable recovery attempts for failed episodes.
    *   `recovery_attempts`: Number of recovery attempts for each failed main episode.

4.  **Output:**
    The generated dataset will be saved as a `.pt` file (e.g., `so101_lerobot_dataset_YYYYMMDD_HHMMSS.pt`) in the current working directory.

### For Imitation Learning with LeRobot

The generated `.pt` file contains episodes formatted for use with the LeRobot library. You can load this dataset and use it to train imitation learning models. Refer to the `so101_il_lerobot_format.ipynb` notebook for an example of how to work with the `SO101LeRobotWrapper` and the expected data format.


