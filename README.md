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

### Environments

| <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/c7d5ca7b-f225-40b1-bb0e-c13a03827fdd" width="200"/> | <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/78fb7a5f-9f91-4873-8a66-5c2a811a8b76" width="200"/> | <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/14fa1182-069f-4a2b-9ad2-47d7a17889ae" width="200"/> |
|---------------------------|---------------------------|---------------------------|
| <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/6e21f9f2-e7b8-42ac-a7af-a7e50838236a" width="200"/> | <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/b45563b2-ba57-4f90-bcac-bdca552bf9a9" width="200"/> | <img src="https://github.com/raulsteinmetz/turtle-dreamer/assets/85199336/5f4a91ac-c189-423c-bba9-b3f62f8c58f3" width="200"/> |


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

![comparison_stages_episodes_2x3](https://github.com/user-attachments/assets/a78b274d-c8f1-4184-bfb1-5e9ed1e85c9a)

Learning curves plots for 360 distance readings as input:

![comparison_stages_episodes_2x3](https://github.com/user-attachments/assets/bf01e62f-6954-48e7-bdcb-674ad3869550)



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
