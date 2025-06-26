# ALOHA Description

Requires MuJoCo 3.1.1 or later.

## Overview

This package contains a simplified robot description (MJCF) of the bimanual
[ALOHA 2](https://aloha-2.github.io/) robot. It is ported from
https://github.com/google-deepmind/mujoco_menagerie/commits/main/ SHA
237f8493d72cd3dc4fb93dbad3745cb55f53f2ef.

The reason for the copy is that some teleop data has been collected on this
version of the robot model. The continuous improvement to the robot model might
break the data. The files in this directory freezes the robot model version.

## License

This model is released under a [BSD-3-Clause License](LICENSE).

## Publications

If you use this model in your work, please use the following citation:

```bibtex
@misc{aloha2_2024,
    title = {ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation},
    url = {https://aloha-2.github.io/},
    author = {ALOHA 2 Team},
    year = {2024},
}
```
