import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        """
        Initialize the CriticNetwork.

        Args:
            beta (float): Learning rate for the optimizer.
            input_dims (int): The shape of the input dimensions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of actions in the action space.
            name (str): The name of the network.
            chkpt_dir (str): Directory where checkpoints are saved.
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self._init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """
        Initialize weights and biases of the network layers.
        """
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight, -f3, f3)
        nn.init.uniform_(self.q.bias, -f3, f3)

        f4 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        nn.init.uniform_(self.action_value.weight, -f4, f4)
        nn.init.uniform_(self.action_value.bias, -f4, f4)

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state: The state input.
            action: The action input.

        Returns:
            The Q-value output.
        """
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        """
        Save the current state of the network as a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the network state from the checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Save the current state of the network as a 'best' checkpoint.
        """
        print('... saving best checkpoint ...')
        best_checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), best_checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        """
        Initialize the ActorNetwork.

        Args:
            alpha (float): Learning rate for the optimizer.
            input_dims (int): The shape of the input dimensions.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of actions in the action space.
            name (str): The name of the network.
            chkpt_dir (str): Directory where checkpoints are saved.
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)


        self._init_weights()


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        """
        Initialize weights and biases of the network layers.
        """
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.mu.weight, -f3, f3)
        nn.init.uniform_(self.mu.bias, -f3, f3)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: The state input.

        Returns:
            The action probabilities.
        """
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        """
        Save the current state of the network as a checkpoint.
        """
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        Load the network state from the checkpoint.
        """
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        """
        Save the current state of the network as a 'best' checkpoint.
        """
        print('... saving best checkpoint ...')
        best_checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), best_checkpoint_file)