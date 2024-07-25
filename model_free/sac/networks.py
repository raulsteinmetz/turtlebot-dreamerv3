import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=400, fc2_dims=300, name='critic', chkpt_dir='tmp/sac'):
        """
        Initialize the CriticNetwork.

        Args:
            beta (float): Learning rate for the optimizer.
            input_dims (int): Dimensions of the input state.
            n_actions (int): Number of possible actions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            name (str): Name of the network for checkpointing.
            chkpt_dir (str): Directory for saving checkpoints.
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')


        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state: The state input.
            action: The action input.

        Returns:
            The Q-value of the state-action pair.
        """
        action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        action_value = F.relu(self.fc2(action_value))
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        """
        Save the current state of the network as a checkpoint.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the network state from the checkpoint.
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=400, fc2_dims=300, name='value', chkpt_dir='tmp/sac'):
        """
        Initialize the ValueNetwork.

        Args:
            beta (float): Learning rate for the optimizer.
            input_dims (int): Input dimensions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            name (str): Name of the network for checkpointing.
            chkpt_dir (str): Directory for saving checkpoints.
        """
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input state.

        Returns:
            The value of the given state.
        """
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        """
        Save the current state of the network as a checkpoint.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the network state from the checkpoint.
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=400, fc2_dims=300, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        """
        Initialize the ActorNetwork.

        Args:
            alpha (float): Learning rate for the optimizer.
            input_dims (int): Input dimensions.
            max_action (float): Maximum action value.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of actions.
            name (str): Name of the network for checkpointing.
            chkpt_dir (str): Directory for saving checkpoints.
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass to compute the mean (mu) and standard deviation (sigma) of the action distribution.

        Args:
            state: Input state.

        Returns:
            mu: Mean of the action distribution.
            sigma: Standard deviation of the action distribution.
        """
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        """
        Sample an action from the normal distribution, using reparameterization if needed.

        Args:
            state: Input state.
            reparameterize (bool): Whether to use reparameterization trick.

        Returns:
            action: Sampled action.
            log_probs: Log probability of the action.
        """
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        actions = probabilities.rsample() if reparameterize else probabilities.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        """
        Save the current state of the network as a checkpoint.
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the network state from the checkpoint.
        """
        self.load_state_dict(T.load(self.checkpoint_file))