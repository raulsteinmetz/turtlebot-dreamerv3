import argparse
import os
import shutil

def copy(fpath1: str, fpath2: str):
    os.makedirs(fpath2, exist_ok=True)
    
    for file in os.listdir(fpath1):
        source_file = os.path.join(fpath1, file)
        destination_file = os.path.join(fpath2, file)

        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy models and logs into the best_models folder')
    parser.add_argument('--agent', type=str, default='sac', help='Specify the RL agent (sac, ddpg, td3, sac_x_hybrid, sac_x)')
    parser.add_argument('--stage', type=int, default=1, help='Specify the environment stage: 1, 2, 3, 4')
    parser.add_argument('--lidar', type=int, default=0, help='Specify the number of LIDAR readings: 10, 360')
    args = parser.parse_args()

    fpath1 = f'models/{args.agent}/stage{args.stage}/'
    fpath2 = f'best_models/lidar{str(args.lidar)}/{args.agent}/stage{args.stage}/'

    copy(fpath1, fpath2)
