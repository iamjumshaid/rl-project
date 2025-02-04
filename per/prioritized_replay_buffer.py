import torch
import random
from per.segment_tree import MinSegmentTree, SumSegmentTree


class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, alpha: float):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        """
        # Maximum priority for storing new samples
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

        :param obs: The current observation.
        :param action: The action.
        :param reward: The reward.
        :param next_obs: The next observation.
        :param terminated: Whether the episode terminated.
        """
        idx = self.pointer

        transition = (obs, action, reward, next_obs, terminated)
        self.buffer[idx] = transition

        scaled_priority = self.max_priority**self.alpha
        self.sum_tree[idx] = scaled_priority
        self.min_tree[idx] = scaled_priority

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

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

    def sample(self, batch_size: int, beta: float) -> torch.Tensor:
        """
        Sample a batch of transitions uniformly and with replacement. The respective elements e.g. states, actions, rewards etc. are stacked

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), where each tensors is stacked.
        """
        assert len(self) >= batch_size

        indices = self._sample_indices_by_priority(batch_size)
        weights = self._compute_importance_weights(indices, beta)
        transitions = [self.buffer[idx] for idx in indices]

        obs, actions, rewards, next_obs, terminated = zip(*transitions)

        # Convert to torch tensors and stack
        obs = torch.stack([o.clone().detach() for o in obs])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_obs = torch.stack([no.clone().detach() for no in next_obs])
        terminated = torch.tensor(terminated)
        weights = torch.tensor(weights)

        return obs, actions, rewards, next_obs, terminated, indices, weights
