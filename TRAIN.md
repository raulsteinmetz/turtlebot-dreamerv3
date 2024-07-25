## Training Your Agent
- Run the gazebo simulation:

  `ros2 launch turtlebot3_gazebo turtle_stage<number>.py`

- To train the model-free agent:
  
   `./train.sh <agent> <stage> <distance_readings(input for drl alg)>`
  
- To train TurtleDreamer (Dreamerv3 for this task):
  
   ```
   cd dreamerv3-torch
   python3 dreamer.py --configs turtle --task turtle --logdir ./logdir/turtle
   ```