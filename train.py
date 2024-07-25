import rclpy
import argparse
import yaml
import time
import numpy as np
import pandas as pd
import torch
import random

from tools import make_agent, make_env

STEP_SYNC = 0.15

configs = {}

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(agent, env):
    if configs['load_models']:
        agent.load_models()

    acum_rwds = []
    steps_rwds = []
    mov_avg_rwds = []

    save_csv_every = 100
    best_moving_average = -np.inf
    n_steps = 0

    print(f'Traning for {configs["train_episodes"]} episodes')
    for episode in range(configs['train_episodes']):
        step = 0
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
            _ = agent.learn()


            step += 1
            n_steps += 1

            step_finish = time.time()
            elapsed = step_finish - step_start
            time.sleep(STEP_SYNC - elapsed if elapsed < STEP_SYNC else 0)  # syncthing with dreamer step freq

        
        print(f"Episode {episode} * Accumulated Reward is ==> {acum_reward}")

        acum_rwds.append(acum_reward)
        steps_rwds.append(n_steps)

        if episode >= save_csv_every - 1:
            moving_avg = np.mean(acum_rwds[-save_csv_every:])
            mov_avg_rwds.append(moving_avg)

            if moving_avg > best_moving_average:
                best_moving_average = moving_avg
                agent.save_models()
                print(f"Saving best models with moving average reward {best_moving_average}...")
        else:
            mov_avg_rwds.append(np.mean(acum_rwds[:episode + 1]))

        if episode == 1 or episode % 100 == 0:
            df = pd.DataFrame({'scores': acum_rwds, 'steps': steps_rwds})
            df.index.name = 'episode'
            fpath = agent.checkpoint_dir
            df.to_csv(f'{fpath}/train.csv')

            
    df = pd.DataFrame({'scores': acum_rwds, 'steps': steps_rwds})
    df.index.name = 'episode'
    fpath = agent.checkpoint_dir
    df.to_csv(f'{fpath}/train.csv')


    return acum_rwds, steps_rwds




def main(args=None):
    rclpy.init()
    env = make_env(configs['stage'], configs['max_steps_per_episode'], configs['lidar'])
    agent = make_agent(env, configs)
    
    _, _ = train(
        agent=agent,
        env=env
        )
    
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Agent on TurtleBot3 Navigation Environment')
    parser.add_argument('--agent', type=str, default='sac', help='Specify the RL agent (sac, ddpg, td3, sac_x_hybrid, sac_x)')
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--load', type=bool, default=False, help='Load agent network models: True, False')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    configs = {}

    configs['agent'] = args.agent
    configs['stage'] = args.stage
    configs['load_models'] = args.load
    configs['test'] = False
    configs['lidar'] = args.lidar

    with open('configs.yaml', 'r') as file:
        config_data = yaml.safe_load(file)

        if args.agent in config_data:
            configs.update(config_data[args.agent])
            configs['train_episodes'] = config_data[args.agent].get('train_episodes', 5001)
            configs['max_steps_per_episode'] = config_data[args.agent].get('max_steps_per_episode', 500) # 500 for stage 6
        else:
            raise ValueError(f"No configuration found for agent: {args.agent}")

    set_all_seeds()
    main()