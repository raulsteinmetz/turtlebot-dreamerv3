import numpy as np

class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: int, n_actions: int):
        """
        Initialize the ReplayBuffer.

        Args:
            max_size (int): The maximum size of the memory buffer.
            input_shape (int): The shape of the input states.
            n_actions (int): The number of actions.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a transition in the memory buffer.

        Args:
            state: The starting state.
            action: The action taken.
            reward: The reward received.
            state_: The resulting state.
            done: Boolean indicating whether the episode is finished.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A tuple containing states, actions, rewards, new states, and done flags.
        """
        if self.mem_cntr < batch_size:
            return None  # Not enough samples to provide a full batch

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones