import random
import heapq
import torch
import torch.nn.functional as F


class PrioritizedReplayBufferHeap:
    def __init__(self, max_size: int, alpha: float = 0.2, beta: float = 0.4):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.pointer = 0
        self.size = 0
        self.max_priority = 1.0

        # Experience buffer
        self.buffer = [None] * max_size

        # Min-heap for storing (priority, index) tuples
        self.priority_heap = []

    def __len__(self):
        return self.size

    def store(self, obs, action, reward, next_obs, terminated):
        idx = self.pointer
        transition = (obs, action, reward, next_obs, terminated)
        self.buffer[idx] = transition

        # Assign priority to new transition
        priority = self.max_priority**self.alpha
        heapq.heappush(self.priority_heap, (priority, idx))

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_experience_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            new_priority = priority**self.alpha
            heapq.heappush(self.priority_heap, (new_priority, idx))
            self.max_priority = max(self.max_priority, new_priority)

    def _sample_indices(self, batch_size):
        """Sample indices based on softmax priority from a subset of 1000 randomly chosen elements."""
        sample_size = min(
            1000, len(self)
        )  # Ensure we don't sample more than available elements
        subset = random.sample(
            self.priority_heap[-len(self) :], sample_size
        )  # Randomly sample 1k elements

        priorities, indices = zip(*subset)  # Extract priorities and indices
        priorities = torch.tensor(priorities, dtype=torch.float32)

        # Compute softmax probabilities
        probs = F.softmax(priorities, dim=0).numpy()
        sampled_indices = random.choices(indices, weights=probs, k=batch_size)

        return sampled_indices, priorities, probs

    def _compute_importance_weights(self, indices, priorities, probs):
        """Calculate importance-sampling weights."""
        p_min = min(priorities) / sum(priorities)
        max_weight = (p_min * len(self)) ** -self.beta
        weights = (
            (torch.tensor(probs[: len(indices)]) * len(self)) ** -self.beta
        ) / max_weight
        return weights

    def sample(self, batch_size, beta):
        assert len(self) >= batch_size

        indices, priorities, probs = self._sample_indices(batch_size)
        weights = self._compute_importance_weights(indices, priorities, probs)
        transitions = [self.buffer[idx] for idx in indices]
        obs, actions, rewards, next_obs, terminated = zip(*transitions)

        obs = torch.stack([o.clone().detach() for o in obs])
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_obs = torch.stack([no.clone().detach() for no in next_obs])
        terminated = torch.tensor(terminated)
        weights = torch.tensor(weights)

        return obs, actions, rewards, next_obs, terminated, indices, weights
