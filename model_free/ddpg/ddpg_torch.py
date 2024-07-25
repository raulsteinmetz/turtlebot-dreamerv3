import os
import numpy as np
import torch as T
import torch.nn.functional as F
from model_free.ddpg.networks import ActorNetwork, CriticNetwork
from model_free.ddpg.noise import OUActionNoise
from model_free.util.buffer import ReplayBuffer

class Agent:
    def __init__(self, alpha=0.0001, beta=0.001, tau=0.001, n_actions=0, input_dims=0,
                 gamma=0.99, max_size=100000, fc1_dims=150, fc2_dims=256, 
                 batch_size=128, max_action=0, min_action=0, checkpoint_dir='tmp/ddpg'):
        """
        Initialize the Agent.

        Args:
            alpha (float): Learning rate for the actor.
            beta (float): Learning rate for the critic.
            tau (float): Soft update parameter.
            n_actions (int): Number of actions.
            input_dims (int): Input dimensions.
            gamma (float): Discount factor.
            max_size (int): Maximum size of the replay buffer.
            fc1_dims (int): Dimension of the first fully connected layer.
            fc2_dims (int): Dimension of the second fully connected layer.
            batch_size (int): Batch size for training.
            max_action (float): Maximum action value.
            min_action (float): Minimum action value.
            checkpoint_dir (str): Directory for saving checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.name = 'ddpg'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, 'actor', checkpoint_dir).to(self.device)
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'critic', checkpoint_dir).to(self.device)
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, 'target_actor', checkpoint_dir).to(self.device)
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, 'target_critic', checkpoint_dir).to(self.device)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        Choose an action based on the current observation using the actor network.

        Args:
            observation: The current state observation.

        Returns:
            The action chosen by the actor network with added noise for exploration.
        """
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]
        # maybe multiplying the returned action by 0.5 will diminish the erratic behaviour

    def remember(self, state, action, reward, state_, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: The starting state.
            action: The action taken.
            reward: The reward received.
            state_: The resulting state.
            done: Boolean indicating whether the episode is finished.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        """
        Save the current state of all networks (actor and critic) as checkpoints.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """
        Load the saved states of all networks (actor and critic) from checkpoints.
        """
        print(f'... loading models ... -> {self.checkpoint_dir}')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        target = rewards + self.gamma * critic_value_.view(-1)
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -T.mean(self.critic.forward(states, self.actor.forward(states)))
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update the target networks
        with T.no_grad():
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)