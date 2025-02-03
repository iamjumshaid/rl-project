import torch
import random
from integrated_agent.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, num_steps: int, gamma: float, alpha: float):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        :param num_steps: Number of steps for multi-step learning.
        :param gamma: Discount factor for future rewards.
        """
        self.max_size = max_size
        self.alpha = alpha
        self.max_priority = 1.0
        self.pointer = 0
        self.size = 0

        self.buffer = [None] * max_size

        # capacity must be positive and a power of 2.
        tree_size = 1
        while tree_size < self.max_size:
            tree_size *= 2

        self.min_tree = MinSegmentTree(tree_size)
        self.sum_tree = SumSegmentTree(tree_size)

        self.num_steps = num_steps
        self.gamma = gamma

    def __len__(self) -> int:
        """Returns how many transitions are currently in the buffer."""
        return self.size

    def store(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminated: torch.Tensor,
    ):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.
        """
        idx = self.pointer

        transition = (obs, action, reward, next_obs, terminated)
        self.buffer[idx] = transition

        scaled_priority = self.max_priority**self.alpha
        self.sum_tree[idx] = scaled_priority
        self.min_tree[idx] = scaled_priority

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, beta: float):
        """
        Sample a batch of transitions using multi-step returns.

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, steps_batch)
                  where reward_batch is the multi-step return and steps_batch contains the actual number of steps used.
        """
        if len(self) < self.num_steps:
            return None
        assert len(self) >= batch_size

        indices = self._sample_indices_by_priority(batch_size)
        weights_batch = self._compute_importance_weights(indices, beta)

        (
            obs_batch,
            action_batch,
            reward_batch,
            next_obs_batch,
            terminated_batch,
            steps_batch,
        ) = ([], [], [], [], [], [])

        for index in indices:
            R_t_n = 0
            actual_steps = 0
            for k in range(self.num_steps):
                if index + k >= self.pointer:  # Avoid out-of-bounds access
                    break

                R_t_n += self.gamma**k * self.buffer[index + k][2]
                actual_steps += 1
                # If a terminal state is encountered, stop the accumulation.
                if self.buffer[index + k][-1]:
                    break

            # Use the first transition for the observation and action,
            # and the last transition actually used for next_obs and termination flag.
            obs, action, _, _, _ = self.buffer[index]
            _, _, _, next_obs, terminated = self.buffer[index + actual_steps - 1]

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
            torch.tensor(steps_batch, dtype=torch.float32),
            indices,
            torch.tensor(weights_batch),
        )

    def update_experience_priorities(
        self, indices: list[int], priorities: torch.Tensor
    ):
        for idx, priority in zip(indices, priorities):
            scaled_priority = priority**self.alpha

            self.sum_tree[idx] = scaled_priority
            self.min_tree[idx] = scaled_priority

            self.max_priority = max(scaled_priority, self.max_priority)

    def _sample_indices_by_priority(self, batch_size):
        """Sample indices proportional to priorities."""
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment_len = p_total / batch_size

        indices = []
        for i in range(batch_size):
            segment_start = segment_len * i
            segment_end = segment_len * (i + 1)
            sample_idx = self.sum_tree.retrieve(
                random.uniform(segment_start, segment_end)
            )
            indices.append(sample_idx)

        return indices

    def _compute_importance_weights(self, indices: list[int], beta: float):
        """Calculate importance-sampling weights for sampled indices."""
        p_total = self.sum_tree.sum()
        p_min = self.min_tree.min() / p_total

        max_weight = (p_min * len(self)) ** -beta

        weights = []
        for idx in indices:
            weight = ((self.sum_tree[idx] / p_total) * len(self)) ** -beta / max_weight
            weights.append(weight)

        return weights
