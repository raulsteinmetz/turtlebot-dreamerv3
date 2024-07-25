# Encoded Representations and World Modeling <br> for Autonomous Terrestrial Robot Navigation

Deep reinforcement learning algorithms have proven effective in handling autonomous navigation
for terrestrial robots using distance readings as environmental observations. However, existing 
architectures are limited to processing small numbers of distance readings, typically 10 equally 
spaced readings over a 360-degree range. This simplification can cause the agent to miss 
information on small and distant objects, leading to collisions. Additionally, these 
architectures are all model-free and do not take advantage of a transition model of the 
environment, which can reduce training time and improve performance. We explore the DreamerV3 
framework for the navigation of terrestrial mobile robots using distance readings as input. 
The framework leverages model-based reinforcement learning through the world modeling paradigm, 
and integrates a variational autoencoder to process observations, allowing for the efficient 
handling of large vectors of distance readings through policy-independent representation learning, 
thereby enhancing navigation capabilities. We evaluate the algorithm using the simulated TurtleBot 
benchmark, extending traditional environments to include long-distance and obstacle-filled navigation 
scenarios. Our approach is compared against established algorithms in the task and shows superior 
performance in most scenarios with both full (360) and reduced (10) lidar readings, the latter being 
the standard in previous works. Notably, our approach achieves 100\% completion in every tested 
environment with the full 360 readings, a performance unmatched by previous works.

### Architecture

#### World Model Traning

![world_model_learning-1](https://github.com/user-attachments/assets/57c8696e-b12c-4797-8cfc-95be21733852)


#### Actor Critic Learning

![actor_critic_learning-1](https://github.com/user-attachments/assets/7e0c4fd3-f8f2-4f19-b4b3-997e1ff5080e)

### Environments

![environments-1](https://github.com/user-attachments/assets/d41377b8-77c8-4ccf-93ef-956a2742f8d1)


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
