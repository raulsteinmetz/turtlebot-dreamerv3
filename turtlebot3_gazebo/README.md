## Configuring ROS packages

To configure the Deep Reinforcement Learning (DRL) stages on gazebo, follow these steps:

1. **Install a ROS2 distribution (foxy and humble are recommended)**
 
   [Follow this tutorial for foxy](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
   
   [Follow this tutorial for humble](https://docs.ros.org/en/humble/Installation.html)
   

2. **Configure robotis gazebo package**

   [Follow this tutorial](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

2. **Create a backup folder for your old `launch`, `models`, and `worlds` directories**

   ```bash
   mkdir ~/backup
   ```

4. **Copy the contents of the `launch`, `models`, and `worlds` directories into your robotis turtlebot3 environment:**

   - For `launch`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch/ ~/backup/
     cp -r turtlebot3_gazebo/launch ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/launch
     ```
   
   - For `models`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models/ ~/backup/
     cp -r turtlebot3_gazebo/models ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/models
     ```
   
   - For `worlds`:
     ```bash
     mv ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/ ~/backup
     cp -r turtlebot3_gazebo/worlds ~/your_ros_distro/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
     ```

5. **Build your project to apply the changes**
   ```bash
   cd ~/your_ros_distro/
   colcon build --symlink-install