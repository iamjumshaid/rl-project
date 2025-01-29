import torch
import random

class MultiStepReplayBuffer:
    def __init__(self, max_size: int, num_steps: int, gamma: float):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        :param n: Number of steps for multi-step learning.
        :param gamma: Discount factor for future rewards.
        """
        self.data = []
        self.max_size = max_size
        self.position = 0

        self.num_steps = num_steps  # Number of steps to look ahead
        self.gamma = gamma  # Discount factor for multistep

    def __len__(self) -> int:
        """Returns how many transitions are currently in the buffer."""
        return len(self.data)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.

        :param obs: The current observation.
        :param action: The action.
        :param reward: The reward.
        :param next_obs: The next observation.
        :param terminated: Whether the episode terminated.
        """
        transition = (obs, action, reward, next_obs, terminated)
        self.data.append(transition)

        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample_multi_step(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of transitions using multi-step returns.

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch) with multi-step returns.
        """

        # Prevent sampling when buffer is too small
        if len(self.data) < self.num_steps + 1:
            return None

        indices = random.choices(range(len(self.data) - self.num_steps), k=batch_size)  # Ensures we have 'n' steps ahead

        obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch = [], [], [], [], []

        for index in indices:
            # Compute n-step return: R_t_n = r_t + γ * r_t+1 + γ² * r_t+2 + ... + γⁿ⁻¹ * r_t+n⁻¹
            R_t_n = sum(self.gamma ** k * self.data[index + k][2] for k in range(self.num_steps))

            obs, action, _, _, _ = self.data[index]  # Get initial observation and action
            _, _, _, next_obs, terminated = self.data[index + self.num_steps - 1]  # Get n-step next state and terminal flag

            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(R_t_n)  # Store computed multi-step return
            next_obs_batch.append(next_obs)
            terminated_batch.append(terminated)

        return (
            torch.stack(obs_batch),
            torch.tensor(action_batch, dtype=torch.long),
            torch.tensor(reward_batch, dtype=torch.float32),
            torch.stack(next_obs_batch),
            torch.tensor(terminated_batch, dtype=torch.float32),
        )
