import argparse
import numpy as np
import os
import pandas as pd

def build_csv(fpath: str, save_to: str):
    files = [f for f in os.listdir(fpath) if f.endswith('.npz')]
    files.sort()

    rewards = []
    steps = []
    current_step = 0

    for file in files:
        file_path = os.path.join(fpath, file)
        data = np.load(file_path)
        if 'reward' in data:
            for reward in data['reward']:
                if reward == -10 or reward == 100:
                    rewards.append(reward)
                    steps.append(current_step)
                current_step += 1

    reward_df = pd.DataFrame({
        'scores': rewards,
        'steps': steps
    })
    
    reward_df['episode'] = range(1, len(reward_df) + 1)

    reward_df.to_csv(save_to, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build dreamer train csv from .npz's")
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    fpath = f'./dreamerv3-torch/logdir/stage{args.stage}_{args.lidar}/train_eps'
    save_to = f'./best_models/lidar{args.lidar}/dreamer/stage{args.stage}/train.csv'
    build_csv(fpath, save_to)
