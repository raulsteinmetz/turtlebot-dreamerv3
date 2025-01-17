# World Models for Autonomous Terrestrial Robot Navigation

Reinforcement learning (RL) algorithms have proven to be highly effective for au-
tonomous navigation of terrestrial robots using distance readings from infrared sen-
sors as environment observations. However, current RL architectures for mobile nav-
igation do not separate the processing of observations from policy learning, which
limits their ability to handle large sets of sensor readings due to the extensive pa-
rameter search space required for the policy network. This limitation forces the use
of a small sample of the original sensor readings as observation, providing a poor
description of the environment, increasing collision risk, and reducing navigation per-
formance. Additionally, these architectures are model-free, missing the advantages
of using a transition dynamics model that could improve decision-making. In this
study, we present a new architecture for terrestrial robot navigation using distance
readings, based on the DreamerV3 algorithm, which makes use of world modeling
through an environment dynamics predictor and an autoencoder for observation
processing. Our approach was tested on the TurtleBot3 robot in simulation and
outperformed traditional algorithms with both full and reduced distance readings.
Notably, it achieved 100% completion in all tested environments with full 360-degree
readings.


### Results

Success rate on tests over 100 episodes in the turtlebot navigation stages for 10 distance sensor readings.

| **Algorithm**                  | DDPG   | SAC    | TD3    | **DreamerV3** |
|--------------------------------|--------|--------|--------|--------------------------|
| **Stage 1**                    | 100.00%| 99.00% | 100.00%| **100.00%**              |
| **Stage 2**                    | 77.00% | 89.00% | 85.00% | **99.00%**               |
| **Stage 3**                    | 99.00% | 96.00% | **100.00%** | 99.00%           |
| **Stage 4**                    | 89.00% | 95.00% | 94.00% | **96.00%**               |
| **Stage 5**                    | 90.00% | 84.00% | **93.00%** | 87.00%           |
| **Stage 6**                    | 71.00% | 56.00% | 75.00% | **80.00%**               |
| **Mean**                       | 87.67% | 86.50% | 91.17% | **93.50%**               |
| **Median**                     | 89.50% | 92.00% | 93.50% | **97.50%**               |

Success rate on tests over 100 episodes in the turtlebot navigation stages for 360 distance sensor readings.

| **Algorithm**                  | DDPG   | SAC    | TD3    | **DreamerV3** |
|--------------------------------|--------|--------|--------|--------------------------|
| **Stage 1**                    | 96.00% | **100.00%** | 18.00% | **100.00%**          |
| **Stage 2**                    | 12.00% | 80.00% | 17.00% | **100.00%**               |
| **Stage 3**                    | **100.00%** | **100.00%** | 7.00% | **100.00%**          |
| **Stage 4**                    | 10.00% | 54.00% | 19.00% | **100.00%**               |
| **Stage 5**                    | 4.00%  | 79.00% | 4.00%  | **100.00%**               |
| **Stage 6**                    | 0.00%  | 92.00% | 1.00%  | **100.00%**               |
| **Mean**                       | 37.00% | 84.17% | 11.00% | **100.00%**               |
| **Median**                     | 11.00% | 86.00% | 12.00% | **100.00%**               |



Learning curves for 10 distance readings as input:

![comparison_stages_episodes_2x3](https://github.com/user-attachments/assets/fbeba89e-9ff4-43e5-b2ab-d0c93a6138a3)


Learning curves plots for 360 distance readings as input:

![comparison_stages_episodes_2x3](https://github.com/user-attachments/assets/8f0abf10-3f8a-4ca4-9d01-ac6ca308bef5)


### Configuring ROS packages
Refer to `./turtlebot3_gazebo/README.md`
  
### Training Your Agent
Refer to `./TRAIN.md`

### Folder Structure
- `./best_models` contains train and test information from algorithms and trained models
- `./dreamerv3-torch` holds code for the DreamerV3 algorithm
- `./model_free` holds the implementations of sac, ddpg and td3
- `./models` keeps models and logs while the algorithms are training
- `./plots` keeps comparisson plots in pdf format
-  `./turtle_env` keeps the gym code for the TurtleBot3 task
-  `./turtlebot3_gazebo` keeps the ROS2 files needed for configuration  

### The code on this repository was inspired by:

- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
- [dranaju/project](https://github.com/dranaju/project)
