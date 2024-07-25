import os
import torch as T
import torch.nn.functional as F
import numpy as np
from model_free.util.buffer import ReplayBuffer
from model_free.sac.networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=0, max_action=0, 
                 gamma=0.99, n_actions=2, max_size=100000, tau=0.001, batch_size=128,
                 reward_scale=2, min_action=0, checkpoint_dir='tmp/sac'):
        """
        Initialize the Soft Actor-Critic agent.

        Args:
            alpha (float): Learning rate for the actor network.
            beta (float): Learning rate for the critic and value networks.
            input_dims (int): Dimensions of the input state.
            max_action (float): Maximum magnitude of action.
            gamma (float): Discount factor for future rewards.
            n_actions (int): Number of actions.
            max_size (int): Maximum size of the replay buffer.
            tau (float): Soft update coefficient for target networks.
            batch_size (int): Size of the batch for learning.
            reward_scale (float): Scaling factor for rewards.
            min_action (float): Minimum magnitude of action.
            checkpoint_dir (str): Directory to save the model checkpoints.
        """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.name = 'sac'
        self.scale = reward_scale
        self.checkpoint_dir = checkpoint_dir

        self.actor = ActorNetwork(alpha, input_dims, max_action=max_action,
                                  n_actions=n_actions, name='actor', chkpt_dir=checkpoint_dir)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1', chkpt_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2', chkpt_dir=checkpoint_dir)
        self.value = ValueNetwork(beta, input_dims, name='value', chkpt_dir=checkpoint_dir)
        self.target_value = ValueNetwork(beta, input_dims, name='target_value', chkpt_dir=checkpoint_dir)

        self.update_network_parameters(tau=1)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def choose_action(self, observation):
        """
        Choose an action based on the current observation.

        Args:
            observation: The current state observation.

        Returns:
            action: The action chosen by the agent.
        """
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

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

    def update_network_parameters(self, tau=None):
        """
        Perform a soft update of the target network parameters.

        Args:
            tau (float, optional): The update factor. If None, use self.tau.
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_dict = dict(value_params)

        for name in value_dict:
            value_dict[name] = tau * value_dict[name].clone() + (1 - tau) * target_value_dict[name].clone()

        self.target_value.load_state_dict(value_dict)

    def save_models(self):
        """
        Save the current state of all networks as checkpoints.
        """
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        """
        Load the saved states of all networks from checkpoints.
        """
        print(f'... loading models ... -> {self.checkpoint_dir}')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        """
        Train the agent from a batch of experiences in the replay buffer.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)


        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

        return critic_1_loss