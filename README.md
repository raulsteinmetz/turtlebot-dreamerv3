# World Models for Autonomous Terrestrial Robot Navigation

Reinforcement Learning (RL) algorithms are effective for simple autonomous terrestrial robot navigation tasks using
infrared sensor distance readings as observations. However, current methods are limited as they directly process
sensor data within policy networks, making it challenging to handle large sensor arrays, and do not utilize model-based
approaches, reducing their decision-making capabilities. This study introduces a new model-based RL architecture for
terrestrial robot navigation based on the DreamerV3 algorithm. It employs a world model containing an autoencoder
for efficient sensor data processing and a dynamics predictor for enhanced decision-making. Comparative analysis of
experiments conducted using the Turtlebot3 robot on a simulated setting demonstrates that our architecture effectively
manages larger sensor datasets, significantly improving spatial awareness and navigation performance with both
reduced sets and complete 360-degree sets of distance sensor readings.


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
