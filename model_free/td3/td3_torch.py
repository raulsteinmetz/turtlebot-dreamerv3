import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
from model_free.util.buffer import ReplayBuffer

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        """
        Initialize the Critic Network.

        Args:
            beta (float): Learning rate for the optimizer.
            input_dims (int): Dimensions of the input.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of possible actions.
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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')


        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)


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
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

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

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3'):
        """
        Initialize the Actor Network.

        Args:
            alpha (float): Learning rate for the optimizer.
            input_dims (int): Dimensions of the input state.
            fc1_dims (int): Number of neurons in the first fully connected layer.
            fc2_dims (int): Number of neurons in the second fully connected layer.
            n_actions (int): Number of possible actions.
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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')


        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: The state input.

        Returns:
            mu: The action probabilities as a tensor.
        """
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = T.tanh(self.mu(prob))

        return mu

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


class Agent:
    def __init__(self, alpha=0.001, beta=0.001, input_dims=0.005, tau=0.005, 
                 max_action=1, min_action=-1, gamma=0.99, update_actor_interval=8, 
                 warmup=1000, n_actions=2, max_size=100000, layer1_size=400, 
                 layer2_size=300, batch_size=128, noise=0.1, checkpoint_dir='tmp/td3'):
        """
        Initialize the TD3 Agent.

        Args:
            alpha (float): Learning rate for the actor.
            beta (float): Learning rate for the critic.
            input_dims (int): Input dimensions.
            tau (float): Soft update parameter.
            max_action (float): Maximum action value.
            min_action (float): Minimum action value.
            gamma (float): Discount factor.
            update_actor_interval (int): Interval for updating the actor network.
            warmup (int): Warm-up steps before starting updates.
            n_actions (int): Number of actions.
            max_size (int): Maximum size of the replay buffer.
            layer1_size (int): Size of the first layer in the networks.
            layer2_size (int): Size of the second layer in the networks.
            batch_size (int): Batch size for training.
            noise (float): Noise added to the action for exploration.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.gamma = gamma
        self.tau = tau


        self.max_action = T.tensor(max_action, dtype=T.float32).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        self.min_action = T.tensor(min_action, dtype=T.float32).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))


        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.update_actor_iter = update_actor_interval
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, 'actor', checkpoint_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions, 'critic_1', checkpoint_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions, 'critic_2', checkpoint_dir)
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions, 'target_actor', checkpoint_dir)
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions, 'target_critic_1', checkpoint_dir)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                             layer2_size, n_actions, 'target_critic_2', checkpoint_dir)

        self.noise = noise
        self.update_network_parameters(tau=1)
        self.name = 'td3'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    def choose_action(self, observation):
        """
        Choose an action based on the current state observation.

        Args:
            observation: The current state observation.

        Returns:
            The action chosen by the agent.
        """
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), dtype=T.float).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        """
        Store a transition in the agent's memory.

        Args:
            state: The starting state.
            action: The action taken.
            reward: The reward received.
            new_state: The resulting state after the action.
            done: Boolean indicating whether the episode is finished.
        """
        self.memory.store_transition(state, action, reward, new_state, done)


    def learn(self):
        """
        Train the agent from a batch of experiences in the replay buffer.
        """
        if self.memory.mem_cntr < self.batch_size:
            return 
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action, 
                                self.max_action)
        
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        return actor_loss

    def update_network_parameters(self, tau=None):

        """
        Perform a soft update of the target network parameters.

        Args:
            tau (float, optional): The update factor. If None, use self.tau.
        """

        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        """
        Save the current state of all networks as checkpoints.
        """
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        """
        Load the saved states of all networks from checkpoints.
        """
        print(f'... loading models ... -> {self.checkpoint_dir}')
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
