import torch
import random
from segment_tree import SumSegmentTree, MinSegmentTree

class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, alpha: float = 0.6):
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
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.min_tree = MinSegmentTree(tree_capacity)
        self.sum_tree = SumSegmentTree(tree_capacity)

    def __len__(self) -> int:
        """Returns how many transitions are currently in the buffer."""
        return self.size

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor):
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

        priority = self.max_priority ** self.alpha
        self.sum_tree[idx] = priority
        self.sum_tree[idx] = priority

        self.pointer = (self.tree_ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_priorities(self, indices: list[int], priorities: torch.Tensor):
        for idx, priority in zip(indices, priorities): 
            priority = priority ** self.alpha

            self.sum_tree[idx] = priority
            self.sum_tree[idx] = priority

            self.max_priority = max(priority, self.max_priority)

    def _sample_proportional(self, batch_size):
        """Sample indices proportional to priorities."""
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        return [
            self.sum_tree.retrieve(random.uniform(segment * i, segment * (i + 1)))
            for i in range(batch_size)
        ]
    
    def _calculate_weights(self, indices: list[int], beta: float):
        """Calculate importance-sampling weights for sampled indices."""
        p_total = self.sum_tree.sum()
        p_min = self.min_tree.min() / p_total
        max_weight = (p_min * len(self)) ** -beta

        weights = [
            ((self.sum_tree[idx] / p_total) * len(self)) ** -beta / max_weight
            for idx in indices
        ]
        
        return weights

    def sample(self, batch_size: int, beta: float) -> torch.Tensor:
        """
        Sample a batch of transitions uniformly and with replacement. The respective elements e.g. states, actions, rewards etc. are stacked

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), where each tensors is stacked.
        """
        assert len(self) >= self.batch_size and beta > 0

        indices = self._sample_proportional(batch_size)
        weights = self._calculate_weights(indices)
        transitions = [self.buffer[idx] for idx in indices]

        obs, actions, rewards, next_obs, terminated = zip(*transitions)

        # Convert to torch tensors and stack
        obs = torch.stack([torch.tensor(o) for o in obs])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_obs = torch.stack([torch.tensor(no) for no in next_obs])
        terminated = torch.tensor(terminated)

        return obs, actions, rewards, next_obs, terminated, indices, weights