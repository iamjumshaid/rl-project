import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import itertools
import numpy as np
from collections import namedtuple

from DQN import DQN
from utils import linear_epsilon_decay, make_epsilon_greedy_policy
from per.prioritized_replay_buffer_heap import PrioritizedReplayBufferHeap as PrioritizedReplayBuffer


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def update_dqn(
    q: nn.Module,
    q_target: nn.Module,
    optimizer: optim.Optimizer,
    gamma: float,
    obs: torch.Tensor,
    act: torch.Tensor,
    rew: torch.Tensor,
    next_obs: torch.Tensor,
    tm: torch.Tensor,
    indices: list,
    weights: torch.Tensor,
    memory: PrioritizedReplayBuffer,
    priority_eps: float = 1e-6,
):
    """
    Update the DDQN network for one optimizer step.

    :param q: The DQN network.
    :param q_target: The target DQN network.
    :param optimizer: The optimizer.
    :param gamma: The discount factor.
    :param obs: Batch of current observations.
    :param act: Batch of actions.
    :param rew: Batch of rewards.
    :param next_obs: Batch of next observations.
    :param tm: Batch of termination flags.
    """

    with torch.no_grad():
        # Use q (main network) to select the action
        next_q_values = q(next_obs)
        best_actions = torch.argmax(next_q_values, dim=1, keepdim=True)

        # Use q_target (target network) to evaluate the action
        q_target_values = q_target(next_obs)
        max_q_s_prime = torch.gather(
            q_target_values, dim=1, index=best_actions
        ).squeeze(1)

        # Compute TD target
        td_target = rew + gamma * max_q_s_prime * ~tm

    # Compute current Q-values
    q_s_a = torch.gather(q(obs), dim=1, index=act.unsqueeze(1)).squeeze(1)
    elementwise_loss = F.smooth_l1_loss(q_s_a, td_target, reduction="none")

    # Apply importance sampling weights
    loss = torch.mean(elementwise_loss * weights)

    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities in memory
    td_delta = elementwise_loss.detach().cpu().numpy()
    new_priorities = td_delta + priority_eps
    memory.update_experience_priorities(indices, new_priorities)

    return loss.item()


class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        lr=0.001,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.1,
        schedule_duration=10_000,
        update_freq=100,
        maxlen=100_000,
        alpha=0.2,
        beta=0.6,
        device=None,
        training_start=None
    ):
        """
        Initialize the DQN agent.

        :param env: The environment.
        :param gamma: The discount factor.
        :param lr: The learning rate.
        :param batch_size: Mini batch size.
        :param eps_start: The initial epsilon value.
        :param eps_end: The final epsilon value.
        :param schedule_duration: The duration of the schedule (in timesteps).
        :param update_freq: How often to update the Q target.
        :param max_size: Maximum number of transitions in the buffer.
        """

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_duration = schedule_duration
        self.update_freq = update_freq
        self.beta = beta
        self.device = device
        self.training_start = training_start

        # Initialize the Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(maxlen, alpha)

        # Initialize the Deep Q-Network. Hint: Remember observation_space and action_space
        self.q = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q.to(self.device)

        # Initialize the second Q-Network, the target network. Load the parameters of the first one into the second
        self.q_target = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.to(self.device)

        # Create an ADAM optimizer for the Q-network
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)

    def train(self, num_episodes: int) -> EpisodeStats:
        """
        Train the DQN agent.

        :param num_episodes: Number of episodes to train.
        :returns: The episode statistics.
        """
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0
        epsilon = self.eps_start

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print(
                    f"Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}  Epsilon: {epsilon:.3f}"
                )

            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            for episode_time in itertools.count():
                # Get current epsilon value
                epsilon = linear_epsilon_decay(
                    self.eps_start,
                    self.eps_end,
                    current_timestep,
                    self.schedule_duration,
                )

                # Choose action and execute
                obs = torch.as_tensor(obs).to(self.device)
                action = self.policy(obs.unsqueeze(0).float(), epsilon=epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                # Store sample in the replay buffer
                self.replay_buffer.store(
                    torch.tensor(obs),
                    torch.tensor(action),
                    torch.tensor(reward),
                    torch.tensor(next_obs),
                    torch.tensor(terminated),
                )
                
                if len(self.replay_buffer) < self.training_start:
                    continue

                # Linearly increase beta
                fraction = min(i_episode / num_episodes, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # Sample a mini batch from the replay buffer

                if len(self.replay_buffer) > self.batch_size:
                    (
                        obs_batch,
                        act_batch,
                        rew_batch,
                        next_obs_batch,
                        tm_batch,
                        indices,
                        weights,
                    ) = self.replay_buffer.sample(self.batch_size, self.beta)
                    obs_batch = obs_batch.to(self.device)
                    act_batch = act_batch.to(self.device)
                    rew_batch = rew_batch.to(self.device)
                    next_obs_batch = next_obs_batch.to(self.device)
                    tm_batch = tm_batch.to(self.device)
                    weights = weights.to(self.device)

                    # Update the Q network
                    update_dqn(
                        self.q,
                        self.q_target,
                        self.optimizer,
                        self.gamma,
                        obs_batch.float(),
                        act_batch,
                        rew_batch.float(),
                        next_obs_batch.float(),
                        tm_batch,
                        indices,
                        weights,
                        self.replay_buffer,
                    )

                    # Update the current Q target
                    if current_timestep % self.update_freq == 0:
                        self.q_target.load_state_dict(self.q.state_dict())
                    current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs
        return stats
