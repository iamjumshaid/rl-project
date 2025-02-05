import torch
import random

class ReplayBuffer:
    def __init__(self, max_size: int, num_steps: int, gamma: float):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        :param num_steps: Number of steps for multi-step learning.
        :param gamma: Discount factor for future rewards.
        """
        self.data = []
        self.max_size = max_size
        self.position = 0
        self.num_steps = num_steps
        self.gamma = gamma

    def __len__(self) -> int:
        return len(self.data)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.
        """
        transition = (obs, action, reward, next_obs, terminated)
        self.data.append(transition)
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions using multi-step returns.

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, steps_batch)
                  where reward_batch is the multi-step return and steps_batch contains the actual number of steps used.
        """
        if len(self.data) < self.num_steps:
            return None

        indices = random.choices(range(len(self.data) - self.num_steps), k=batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, steps_batch = [], [], [], [], [], []

        for index in indices:
            R_t_n = 0
            actual_steps = 0
            for k in range(self.num_steps):
                R_t_n += self.gamma ** k * self.data[index + k][2]
                actual_steps += 1
                # If a terminal state is encountered, stop the accumulation.
                if self.data[index + k][-1]:
                    break

            # Use the first transition for the observation and action,
            # and the last transition actually used for next_obs and termination flag.
            obs, action, _, _, _ = self.data[index]
            _, _, _, next_obs, terminated = self.data[index + actual_steps - 1]

            obs_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(R_t_n)
            next_obs_batch.append(next_obs)
            terminated_batch.append(terminated)
            steps_batch.append(actual_steps)

        return (
            torch.stack(obs_batch),
            torch.tensor(action_batch, dtype=torch.long),
            torch.tensor(reward_batch, dtype=torch.float32),
            torch.stack(next_obs_batch),
            torch.tensor(terminated_batch, dtype=torch.float32),
            torch.tensor(steps_batch, dtype=torch.float32)
        )