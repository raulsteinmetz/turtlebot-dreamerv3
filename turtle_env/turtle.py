import rclpy
import random
import math
import numpy as np
import torch as T
import torch.nn.functional as F
from time import sleep
from rclpy.node import Node
from std_srvs.srv import Empty 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, SetEntityState
import time


import gym
from gym import spaces


REACH_TRESHOLD = 0.4
LIDAR_MAX_RANGE = 3.5
COLISION_TRESHOLD = 0.2
EASE_DECAY = 0.005
EASE_BEGIN = 0.75
EASE_MIN = 0.01
MEDIUM_RATE = 0.1

def generate_target_sdf(x, y, z):
        return f"""
        <?xml version='1.0'?>
        <sdf version='1.6'>
        <model name='target_mark'>
            <static>true</static>  <!-- This makes the model static -->
            <pose>{x} {y} {z} 0 0 0</pose>
            <link name='link'>
            <visual name='visual'>
                <geometry>
                <plane><normal>0 0 1</normal><size>0.5 0.5</size></plane>  <!-- Smaller size -->
                </geometry>
                <material>
                <ambient>1 0 0 1</ambient>  <!-- Bright red color for visibility -->
                </material>
            </visual>
            </link>
        </model>
        </sdf>
        """

class Env(Node):
    """
    A class representing the environment in which the TurtleBot operates. It is a ROS node 
    that interacts with the Gazebo simulation environment.
    """
    def __init__(self, stage, max_steps, lidar):
        """
        Initialize the environment node.
        """
        super().__init__("trainer_node")


        # Setup publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.get_entity_state_client = self.create_client(GetEntityState, '/demo/get_entity_state')
        self.set_entity_state_client = self.create_client(SetEntityState, '/demo/set_entity_state')

        # pause for rolout
        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')
        

        self.reset_info()
        self.init_properties(stage, max_steps, lidar)

    def pause_simulation(self):
        """
        Pause the Gazebo simulation.
        """
        try:
            pause_request = Empty.Request()
            future = self.pause_simulation_client.call_async(pause_request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info('Simulation paused successfully.')
            else:
                self.get_logger().error('Failed to pause simulation.')
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def unpause_simulation(self):
        """
        Unpause the Gazebo simulation.
        """
        try:
            unpause_request = Empty.Request()
            future = self.unpause_simulation_client.call_async(unpause_request)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                self.get_logger().info('Simulation unpaused successfully.')
            else:
                self.get_logger().error('Failed to unpause simulation.')
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def reset_info(self):
        """
        Reset the internal state information of the environment.
        """
        self.odom_data = None
        self.scan_data = None

    def init_properties(self, stage, max_steps, lidar):
        """
        Initialize the properties of the environment.
        """
        self.num_states = 14
        self.num_actions = 2
        self.action_upper_bound = .25
        self.action_lower_bound = -.25

        self.step_counter = 0
        self.reset_when_reached = True
        self.reached = False

        self.stage = stage
        self.max_steps = max_steps
        self.lidar = lidar

    def odom_callback(self, msg):
        """
        Callback function for odometry data.

        :param msg: The odometry data message.
        """
        self.odom_data = msg

    def scan_callback(self, msg):
        """
        Callback function for LIDAR scan data.

        :param msg: The LIDAR scan data message.
        """
        self.scan_data = msg

    def get_state(self, linear_vel, angular_vel):
        """
        Get the current state of the environment.

        :param linear_vel: Current linear velocity of the robot.
        :param angular_vel: Current angular velocity of the robot.
        :return: The current state of the environment, robot's x and y coordinates, and LIDAR data.
        """
        self.reset_info()
        rclpy.spin_once(self, timeout_sec=0.5)
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        turtle_x = self.odom_data.pose.pose.position.x
        turtle_y = self.odom_data.pose.pose.position.y


        q = self.odom_data.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        angle_to_target = math.atan2(self.target_y - turtle_y, self.target_x - turtle_x) - yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))

        distance_to_target = math.sqrt((self.target_x - turtle_x) ** 2 + (self.target_y - turtle_y) ** 2)

        lidar_readings = self.scan_data.ranges
        num_samples = self.lidar
        step = (len(lidar_readings) - 1) // (num_samples - 1)
        lidar = [lidar_readings[i * step] if lidar_readings[i * step] != float('inf') else LIDAR_MAX_RANGE for i in range(num_samples)]
        # lidar = lidar_readings.tolist()

        state = lidar + [distance_to_target, angle_to_target, linear_vel, angular_vel]
        
        state = F.tanh(T.tensor(state)).tolist() # trying some normalization
        
        return state, turtle_x, turtle_y, lidar


    def respawn_target(self):
        self.despawn_target_mark()
        self.spawn_target_in_environment()


    def reset(self):
        """
        Reset the simulation environment for a new episode.

        :param stage: The stage of training or simulation scenario.
        :return: The initial state of the environment after reset.
        """


        self.step_counter = 0


        if self.reached == False or (self.reached == True and self.reset_when_reached == True):
            req = Empty.Request()
            while not self.reset_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('Reset service not available, waiting again...')
            self.reset_client.call_async(req)

        if self.reset_when_reached == False:
            self.pause_simulation()
            self.respawn_target()
            self.unpause_simulation()
        else:
            self.respawn_target()



        self.publish_vel(0.0, 0.0)


        self.scan_data = None
        self.odom_data = None
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)
            sleep(0.1)

        state, _, _, _ = self.get_state(0, 0)

        
        return state

    def publish_vel(self, linear_vel, angular_vel):
        """
        Publish velocity commands to the robot.

        :param linear_vel: Linear velocity command.
        :param angular_vel: Angular velocity command.
        """
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def get_reward_and_done(self, turtle_x, turtle_y, target_x, target_y, lidar_32):
        """
        Calculate the reward based on the current state of the environment.

        :param turtle_x: Current x-coordinate of the robot.
        :param turtle_y: Current y-coordinate of the robot.
        :param target_x: x-coordinate of the target.
        :param target_y: y-coordinate of the target.
        :param lidar_32: LIDAR sensor data.
        :param steps: Current step count in the episode.
        :param max_steps: Maximum number of steps in an episode.
        :return: Reward for the current state and a boolean indicating if the episode is done.
        """
        reward = 0
        done = False

        distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2) 

        # reward -= distance

        if distance < REACH_TRESHOLD:
            self.reached = True
            done = True
            reward = 100 # 200 for distance based
            print('[log] Turtlebot3 reached target')
        elif np.min(lidar_32) < COLISION_TRESHOLD:
            self.reached = False
            done = True
            reward = -10
            print('[log] Turtlebot3 colided with object')
        elif self.step_counter >= (self.max_steps - 1):
            self.reached = False
            done = True
            reward = -10
            print('[log] Turtlebot3 reached step limit')

        return reward, done



    def spawn_target_in_environment(self):
        """
        Spawn a target in the environment at a random location depending on the stage of training.

        :param stage: The stage of training, which determines the spawning strategy for the target.
        """

        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')


        self.target_x, self.target_y = self.generate_random_target_position()
        fixed_z = 0.01

        request = SpawnEntity.Request()
        request.name = 'target_mark'
        request.xml = generate_target_sdf(self.target_x, self.target_y, fixed_z)


        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_spawn_result(future, fixed_z)

        sleep(0.5)
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def despawn_target_mark(self):
        """
        Despawn the target mark from the environment.
        """
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DeleteEntity service not available, waiting again...')

        request = DeleteEntity.Request()
        request.name = 'target_mark'
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_despawn_result(future)

    def step(self, action):
        """
        Execute a step in the environment based on the given action.

        :param action: The action to be executed.
        :param step: Current step count in the episode.
        :param max_steps_per_episode: Maximum number of steps in an episode.
        :param stage: Current stage of the training.
        :return: Reward for the step, a boolean indicating if the episode is done, and the next state.
        """
        action = action['action']

        rclpy.spin_once(self, timeout_sec=0.5)
        self.publish_action(action)

        rclpy.spin_once(self, timeout_sec=0.5)

        obs, turtle_x, turtle_y, lidar32 = self.get_state(action[0], action[1])

        reward, done = self.get_reward_and_done(turtle_x, turtle_y, self.target_x, self.target_y, lidar32)

        self.step_counter += 1
        
        return reward, done, obs
    
    def generate_random_target_position(self):
        """
        Generate a random position for the target based on the training stage.

        :param stage: The stage of training.
        :return: A tuple (x, y) representing the target's position.
        """


        if self.stage == 1:
            self.target_x = random.uniform(-1.90, 1.90)
            self.target_y = random.uniform(-1.90, 1.90)
        elif self.stage == 2:
            area = np.random.randint(0, 5)
            if area == 0: 
                self.target_x = random.uniform(-1.90, 1.90)
                self.target_y = random.uniform(-1.5, -1.9) 
            elif area == 1:
                self.target_x = random.uniform(-1.90, 1.90)
                self.target_y = random.uniform(1.5, 1.9) 
            elif area == 2:
                self.target_x = random.uniform(1.5, 1.9)
                self.target_y = random.uniform(-1.90, 1.90)
            elif area == 3:
                self.target_x = random.uniform(-1.5, -1.9)
                self.target_y = random.uniform(-1.90, 1.90)
            elif area == 4:
                self.target_x = random.uniform(-0.7, 0.7)
                self.target_y = random.uniform(-0.7, 0.7)
        elif self.stage == 3:
            points = [(0.5, 1), (0, 0.8), (-.4, 0.5), (-1.5, 1.5),
                        (-1.7, 0), (-1.5, -1.5), (1.7, 0), (1.7, -.8), 
                        (1.7, -1.7), (0.8, -1.5), (0, -1), (-.4, -2),
                        (-1.8, -1.8), (-1.8, -0.5), (-1.8, -1), (-1.8, -1.5),
                        (-1.6, -1.6), (-1.5, -1.5), (-1.2, -1.2), (-1.3, -1.3), 
                        (1.5, 1.5), (1.5, 1), (1, 1), (0, 1), (.7, 1.7), (-1., 1.7),
                        (-1.5, 1.7), (1.5, 0)]
            chosen_point = random.choice(points)
            self.target_x, self.target_y = chosen_point
        elif self.stage == 4:
            points = [
                (0.7, 0), (0.7, 0.5), (0.7, 1.0), (0.7, 1.5), (0.7, 2.0),
                (0.5, -0.5), (0.5, -1.), (0.5, -1.5), (0.5, -2.0),
                (0.0, 1.0), (0.0, -1.0), (0.0, -1.5), (0.0, -2.0),
                (-0.7, 0.0), (-0.7, 0.5), (-0.7, 1.0), (-0.7, -0.5), (-0.7, -1.0),
                (-0.7, 2.0), (-0.2, 2.0), (-2.0, 0), (-2.0, -0.5), (-2.0, 0.5), (-2.0, 1.0),
                (-2.0, 1.5), (-2.0, 2.0), (-2.0, -1.0), (-2.0, -2.0), (-1.5, -2.0), (-1, -2.0),
                (2.0, 2.0), (2.0, 1.5), (2.0, 1.2), (1.6, 2.0), (1.6, 1.5), (1.6, 1.2),
                (2.0, 0.0), (1.5, 0.0), (2.0, -0.5), (1.5, -0.5), (2.0, -1.0), (1.5, -1.0), (2.0, -2.0), (1.5, -2.0), (1.0, -2.0)
            ]
            chosen_point = random.choice(points)
            self.target_x, self.target_y = chosen_point
        elif self.stage == 5:
            points = [
                (3.0, 3.0), (3.0, 2.5), (3.0, 1.5), (3.0, 1.0), (3.0, 0.5), (3.0, 0.0),
                (3.0, -3.0), (3.0, -2.5), (3.0, -2.0), (3.0, -1.5), (3.0, -1.0), (3.0, -0.5),
                (2.5, 3.0), (2.0, 3.0), (1.5, 3.0), (1.0, 3.0), (0.5, 3.0), (0.0, 3.0),
                (-2.5, 3.0), (-2.0, 3.0), (-1.5, 3.0), (-1.0, 3.0), (-0.5, 3.0), (-3.0, 3.0),
                (0.0, 2.5), (0.0, 2), (0.5, 2.5), (0.5, 2), (-0.5, 2.5), (-0.5, 2),
                (1.7, 2), (1.7, 1.5), (1.7, 1.0), (1.7, 0.0), (1.7, -0.5), (1.7, -1.0),
                (2.0, 0.0), (2.0, -0.5), (2.0, -1.0), (2.5, 0.0), (2.5, -0.5), (2.5, -1.0),
                (-2.0, 0.0), (-2.0, -0.5), (-2.0, -1.0), (-2.5, 0.0), (-2.5, -0.5), (-2.5, -1.0),
                (0.0, 1.0), (0.0, -1.0), (0.8, 0.0), (-0.8, 0.0), (1.0, 1.0), (-1.0, 1.0), (0.6, -1.0), (-1.0, -1.0),
                (2.5, -3.0), (2.0, -3.0), (1.5, -3.0), (1.0, -3.0), (0.5, -3.0), (0.0, -3.0), (3.0, -3.0),
                (-2.5, -3.0), (-2.0, -3.0), (-1.5, -3.0), (-1.0, -3.0), (-0.5, -3.0), (-3.0, -3.0),
                (0.5, -1.0), (0.1, -1.0), (0.5, -1.5), (0.1, -1.5), (0.5, -2.5), (0.1, -2.5),
                (1.0, -2.5), (1.5, -2.5), (2.0, -2.5), (2.5, -2.5), (3.0, -2.5)
            ]
            chosen_point = random.choice(points)
            self.target_x, self.target_y = chosen_point
        elif self.stage == 6:
            points = [
                # center
                (0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1),
                (1.5, 0), (0, -1.5), (-1.5, 0), (1.5, -1.5),

                # borders

                # upper
                (6.5, 3), (6.5, 2.5), (6.5, 2.0), (6.5, 1.5), (6.5, 1.0), (6.5, 0.5), (6.5, 0),
                (6.5, -3), (6.5, -2.5), (6.5, -2.0), (6.5, -1.5), (6.5, -1.0), (6.5, -0.5),

                # left
                (6, 3), (6, 2.5), (5.5, 3), (5.5, 2.5), (5, 3), (5, 2.5), (4.5, 3), (4.5, 2.5),
                (4, 3), (4, 2.5), (4, 2), (4, 1.5), (3.5, 3), (3.5, 2.5), (3.5, 2), (3.5, 1.5),
                (3, 3), (3, 2.5), (3, 2), (3, 1.5), (2.5, 3), (2.5, 2.5), (2.5, 2), (2.5, 1.5),
                (2, 3), (2, 2.5), (2, 2), (2, 1.5), (1.5, 3), (1.5, 2.5), (1.5, 2),
                (1, 3), (1, 2.5), (1, 2), (0.5, 3), (0.5, 2.5), (0.5, 2),
                (0, 3), (0, 2.5), (0, 2), (-0.5, 3), (-0.5, 2.5), (-0.5, 2),
                (-1.5, 3), (-1.5, 2.5), (-1.5, 2), (-1.5,  1.5),
                (-2, 3), (-2, 2.5), (-2, 2), (-2,  1.5), (-2.5, 3), (-2.5, 2.5),
                (-3.1, 3), (-3.1, 2.5), (-3.1, 2), (-3.1, 1.5),
                (-3.5, 3), (-3.5, 2.5), (-3.5, 2), (-3.5, 1.5),
                (-4.0, 3), (-4.0, 2.5), (-4.0, 2), (-4.0, 1.5),
                
                # bottom
                (-5.5, 3), (-5.5, 2.5), (-5.5, 2), (-5.5, 1.5), (-5.5, 1),
                (-6, 3), (-6, 2.5), (-6, 2), (-6, 1.5), (-6, 1),
                (-6.5, 3), (-6.5, 2.5), (-6.5, 2), (-6.5, 1.5), (-6.5, 1),
                (-6.5, 0.5), (-6.5, 0), (-6.5, -0.5), (-6.5, -1), (-6.5, -1.5),
                (-6, -1), (-6, -1.5), (-5.5, -1), (-5.5, -1.5), (-6.5, -3),
                (-6, -3), (-5.5, -3), (-5, -3),

                # right
                (-4.5, -3), (-4, -3), (-4.5, -2.5), (-4, -2.5),
                (-4.5, -3.2), (-4, -3.2), (-3.5, -3.2), (-3, -3.2),
                (-2.5, -3.2), (-2, -3.2), (-1.5, -3.2), (-1, -3.2),
                (-0.5, -3.2), (0.0, -3.2), (0.5, -3.2), (1, -3.2),
                (1.5, -3.2), (2.0, -3.2), (1.5, -2.5), (2.0, -2.5),
                (3.5, -3.2), (4, -3.2), (3.5, -2.5), (4.0, -2.5),
                (4.5, -3.2), (5, -3.2), (4.5,  -2.5), (5, -2.5),
                (5.5, -2.5), (5.5, -3.2),

                (4, 0), (3.5, 0), (3, 0), (3, -0.5), (3, -1),

                # conects
                (-1.5, 1.5), (1.5, 1.5), (-0.2, 1.5), (-1,  1.5), (-5.0, 1.5)
            ]
            chosen_point = random.choice(points)
            self.target_x, self.target_y = chosen_point
        elif self.stage == 7:

            points = [

                # first quarter
                (2, 2), (1.5, 2), (1, 2), (0.5, 2),
                (2, 1.5), (2, 1), (2, 0.5),
                (1.5, 1.5), (1.5, 1), (1, 1.5),
                (1, 1), (0.5, 1), (1, 0.5),

                # second quarter
                (2, 0), (2, -0.5), (2, -1.0),
                (1, -1.0), (1, -0.5),
                (2, -2), (1.5, -2), (2, -1.5),
                (2, -1), (1, -2),

                # third quarter
                (-2, 2), (-1.5, 2), (-1, 2), (-0.5, 2),
                (-1, 1.5), (-0.5, 1.5),

                # fourth quarter
                (-2, 0), (-1.5, 0), (-1, 0),
                (-2, -0.5), (-1.5, -0.5), (-1, -0.5),
                (-2, -2), (-1.5, -2), (-1, -2), (-0.5, -2),
                (0.0, -2), (0.0, -1.5)

            ]

            chosen_point = random.choice(points)
            self.target_x, self.target_y = chosen_point

        return self.target_x, self.target_y

    def handle_spawn_result(self, future, fixed_z):
        """
        Handle the result of a spawn entity request.

        :param future: The future object returned from the spawn entity service.
        :param fixed_z: The fixed z-coordinate of the spawned entity.
        """
        if future.result() is not None:
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={self.target_x}, y={self.target_y}, z={fixed_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")

    def handle_despawn_result(self, future):
        """
        Handle the result of a despawn entity request.

        :param future: The future object returned from the delete entity service.
        """
        if future.result() is not None and future.result().success:
            self.get_logger().info("Mark deleted successfully.")
        else:
            self.get_logger().info("No mark to delete or deletion failed.")

    def publish_action(self, action):
        """
        Publish the robot's velocity based on the given action.
        :param action: The action to be executed.
        """
        # linear should be aways positive, the robot does not go backwards
        # angular vel is multiplied by two for better robot control
        linear_vel = np.abs(float(action[0])) * 0.1 # .25
        angular_vel = float(action[1]) * 2 * 0.1 # .25

        self.publish_vel(linear_vel, angular_vel)


class Turtle(gym.Env):
    """ turtlebot environment navigation problem """

    def __init__(self, stage, max_steps, lidar):
        super(Turtle, self).__init__()

        self._env = Env(stage, max_steps, lidar)

        self.observation_space = spaces.Dict({
            'sensor_readings': spaces.Box(low=np.zeros(lidar, dtype=np.float32),
                                    high=np.ones(lidar, dtype=np.float32),
                                    shape=(lidar,),
                                    dtype=np.float32),
            'target': spaces.Box(low=np.zeros(2, dtype=np.float32),
                                    high=np.ones(2, dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32),
            'velocity': spaces.Box(low=np.zeros(2, dtype=np.float32),
                                    high=np.ones(2, dtype=np.float32),
                                    shape=(2,),
                                    dtype=np.float32),
        })


        self.action_space = spaces.Box(low=np.array([0, -1.]), high=np.array([1., 1.]), dtype=np.float32)

    
    def step(self, action):
        reward, done, obs = self._env.step(action)
        return {'sensor_readings': obs[:self._env.lidar], 'target': obs[self._env.lidar:-2], 'velocity': obs[self._env.lidar + 2:], "image": np.zeros((4, 4, 3), dtype=np.int8), 'is_first': False, 'is_last': False, 'is_terminal': done}, reward, done, {'discount': 0.99}

    def reset(self):
        obs = self._env.reset()
        return {'sensor_readings': obs[:self._env.lidar], 'target': obs[self._env.lidar:-2], 'velocity': obs[self._env.lidar + 2:], "image": np.zeros((4, 4, 3)), 'is_first': True, 'is_last': False, 'is_terminal': False}

    def close(self):
        self._env.destroy_node()