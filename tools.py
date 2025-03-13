from turtle_env.turtle import Turtle
from turtle_env.real import Turtle as Real

from model_free.sac.sac_torch import Agent as SAC
from model_free.ddpg.ddpg_torch import Agent as DDPG
from model_free.td3.td3_torch import Agent as TD3


def make_env(stage: int, max_steps: int, lidar: int, real=False):
    if not real:
        return Turtle(stage, max_steps, lidar)
    else:
        return Real(stage, max_steps, lidar)
    

def make_agent(env: Turtle, configs: dict, test: bool = False):
    if configs['agent'] == 'sac':
        agent = SAC(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            max_action=1,
            gamma=configs['gamma'],
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            tau=configs['tau'],
            batch_size=configs['batch_size'],
            reward_scale=configs['reward_scale'],
            min_action=-1,
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage{configs['stage']}" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage{configs['stage']}"
            )

        )
    elif configs['agent'] == 'ddpg':
        agent = DDPG(
            alpha=configs['alpha'],
            beta=configs['beta'],
            tau=configs['tau'],
            n_actions=env.action_space.shape[0],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            gamma=configs['gamma'],
            max_size=configs['max_size'],
            fc1_dims=configs['fc1_dims'],
            fc2_dims=configs['fc2_dims'],
            batch_size=configs['batch_size'],
            max_action=1,
            min_action=-1,
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage{configs['stage']}" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage{configs['stage']}"
            )
        )
    elif configs['agent'] == 'td3':
        agent = TD3(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            tau=configs['tau'],
            max_action=1,
            min_action=-1,
            gamma=configs['gamma'],
            update_actor_interval=configs['update_actor_interval'], # dpu
            warmup=configs['warmup'] if not configs['test'] else 0,
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            layer1_size=configs['layer1_size'],
            layer2_size=configs['layer2_size'],
            batch_size=configs['batch_size'],
            noise=configs['noise'],
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage{configs['stage']}" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage{configs['stage']}"
            )
        )
    else:
        raise('Agent not implemented')
    
    return agent