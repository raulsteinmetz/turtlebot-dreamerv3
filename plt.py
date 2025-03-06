import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

def moving_average_std(data, window_size, std_factor=0.2):
    df_sorted = data.sort_values(by='episode').reset_index(drop=True)
    df_smoothed = df_sorted.rolling(window=window_size, min_periods=1).mean()
    df_std = df_sorted.rolling(window=window_size, min_periods=1).std()
    df_std *= std_factor
    return df_smoothed, df_std

def plot_learning_curves(agents, stages, lidar):
    colormaps = {'ddpg': 'Oranges', 'td3': 'Blues', 'sac': 'Reds', 'dreamer': 'Greens'}
    _lidar = lidar
    
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.2) 
    axs = axs.flatten()
    
    for i, stage in enumerate(stages):
        ax = axs[i]
        
        for agent in agents:
            if int(_lidar) == 0:
                lidar = 360 if agent == 'dreamer' else 10
            fpath = f'best_models/lidar{lidar}/{agent}/stage{stage}/train.csv'
            data = pd.read_csv(fpath)
            
            print(f"Columns in {fpath}: {data.columns.tolist()}")
            
            data['scores'][:10] = 0.0
            data['scores'] = data['scores'].apply(lambda x: 0 if x == -10 else (1 if x == 100 else x))
            
            if stage == 1 and (lidar == 10 or int(_lidar) == 0):
                window = 100
                max_episode = 1000
            else: 
                window = 500
                max_episode = 5000

             

            ma_episodes, std_episodes = moving_average_std(data[['episode', 'scores']], window)
            ma_episodes = ma_episodes[:max_episode]
            std_episodes = std_episodes[:max_episode]
            
            name = agent.upper() if agent != 'dreamer' else 'DreamerV3'
            colormap = get_cmap(colormaps[agent])
            
            if agent == 'dreamer':
                main_alpha = 1.0
                std_alpha = 0.4
                color = 'darkgreen'
            else:
                main_alpha = 1.0
                std_alpha = 0.3
                color = colormap(0.5)
            
            ax.plot(ma_episodes['episode'], ma_episodes['scores'], label=f'{name}', linewidth=2, color=color, alpha=main_alpha)
            ax.fill_between(ma_episodes['episode'], ma_episodes['scores'] - std_episodes['scores'], ma_episodes['scores'] + std_episodes['scores'], alpha=std_alpha, color=color, linewidth=0.5)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Rewards')
        ax.set_title(f'Stage {stage}')
        ax.grid(True)
        ax.set_xlim([0, max_episode])
    
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=12, title='Algorithms', title_fontsize=14, ncol=len(agents))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if int(_lidar) == 0:
        fig.savefig(f'plots/any_lidar/comparison_stages_episodes_2x3.png', format='png', bbox_inches='tight')
    else:
        fig.savefig(f'plots/lidar{lidar}/comparison_stages_episodes_2x3.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot multiple RL agents' training learning curves")
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6], help='Specify the environment stages: e.g., 1 2 3 4')
    parser.add_argument('--lidar', type=int, default=-1, help='Specify the lidar readings: 10 or 360, 0 for dreamer 360 others 10 comparison')
    args = parser.parse_args()

    agents = ['ddpg', 'sac', 'td3', 'dreamer']
    plot_learning_curves(agents, args.stages, args.lidar)
