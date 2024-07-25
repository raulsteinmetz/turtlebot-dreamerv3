import rclpy
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import random
import torch

from tools import make_agent, make_env

STEP_SYNC = 0.15

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

configs = {}

def test(agent, env):
    agent.load_models()

    results = []

    for episode in range(configs['test_episodes']):
        steps = 0
        done = False
        acum_reward = 0
        obs_dict = env.reset()

        obs = obs_dict['sensor_readings'] + obs_dict['target'] + obs_dict['velocity']


        while not done:
            step_start = time.time()
            action_dict = {}
            action_dict['action'] = agent.choose_action(obs)
            obs_dict_, reward, done, _ = env.step(action_dict)
            obs_ = obs_dict_['sensor_readings'] + obs_dict_['target'] + obs_dict_['velocity']
            agent.remember(obs, action_dict['action'], reward, obs_, done)
            obs = obs_
            acum_reward += reward

            steps += 1

            step_finish = time.time()
            elapsed = step_finish - step_start
            time.sleep(STEP_SYNC - elapsed if elapsed < STEP_SYNC else 0) # synching with dreamer step freq

        results.append((acum_reward, steps))

    df_results = pd.DataFrame(results, columns=['reward', 'steps'])
    fpath = agent.checkpoint_dir + '/test.csv'
    df_results.to_csv(fpath, index=False)



def main(args=None):
    rclpy.init()
    env = make_env(configs['stage'], configs['max_steps'], configs['lidar'])
    agent = make_agent(env, configs, test=True)
    
    test(
        agent=agent, 
        env=env
        )
    
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Agent on TurtleBot3 Navigation Environment')
    parser.add_argument('--agent', type=str, default='sac', help='Specify the RL agent (sac, ddpg, td3, sac_x_hybrid, sac_x)')
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--episodes', type=int, default=100, help='Specify the number of test episodes')
    parser.add_argument('--max_steps', type=int, default=300, help='Specify the step limit (for episodes)')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    configs = {}

    configs['agent'] = args.agent
    configs['stage'] = args.stage
    configs['test_episodes'] = args.episodes
    configs['max_steps'] = args.max_steps
    configs['test'] = True
    configs['lidar'] = args.lidar
    

    with open('configs.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

        if args.agent in config_data:
            configs.update(config_data[args.agent])
        else:
            raise ValueError(f"No configuration found for agent: {args.agent}")
        
    set_all_seeds()
    main()