import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import namedtuple
import itertools

from utils import linear_epsilon_decay, make_epsilon_greedy_policy
from dqn import DQN
from replay_buffer import MultiStepReplayBuffer

def update_dqn(
        q: nn.Module,
        q_target: nn.Module,
        optimizer: optim.Optimizer,
        gamma: float,
        n_steps: int,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        tm: torch.Tensor,
    ):
    """
    Update the DQN network for one optimizer step.

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
    # Zero out the gradient
    optimizer.zero_grad()

    # Calculate the TD-Target
    with torch.no_grad():
        td_target = rew + (gamma ** n_steps) * (q_target(next_obs)).max(dim=1)[0] * (1 - tm)

    # Calculate the loss. Hint: Pytorch has the ".gather()" function, which collects values along a specified axis using some specified indexes
    predicted_q = torch.gather(q(obs), dim=1, index=act.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(predicted_q, td_target)

    # Backpropagate the loss and step the optimizer
    loss.backward()
    optimizer.step()

    return loss


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

class DQNAgent:
    def __init__(self,
            env,
            gamma=0.99,
            num_steps=3,
            lr=0.001,
            batch_size=64,
            eps_start=1.0,
            eps_end=0.1,
            schedule_duration=10_000,
            update_freq=100,
            maxlen=100_000,
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
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_duration = schedule_duration
        self.update_freq = update_freq

        # Initialize the Multi-step Replay Buffer
        self.replay_buffer = MultiStepReplayBuffer(maxlen, num_steps, gamma)

        # Initialize the Deep Q-Network. Hint: Remember observation_space and action_space
        self.q = DQN(self.env.observation_space.shape, self.env.action_space.n)

        # Initialize the second Q-Network, the target network. Load the parameters of the first one into the second
        self.q_target = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q_target.load_state_dict(self.q.state_dict())

        # Create an ADAM optimizer for the Q-network
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)

        self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)


    def train(self, num_episodes: int , num_steps: int) -> EpisodeStats:
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
                print(f'Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}  Epsilon: {epsilon:.3f}')

            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            for episode_time in itertools.count():
                epsilon = linear_epsilon_decay(self.eps_start, self.eps_end, current_timestep, self.schedule_duration)

                # Choose action and execute
                action = self.policy(torch.as_tensor(obs).unsqueeze(0).float(), epsilon=epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                self.replay_buffer.store(
                    torch.as_tensor(obs),
                    torch.as_tensor(action),
                    torch.as_tensor(reward),
                    torch.as_tensor(next_obs),
                    torch.as_tensor(terminated)
                )

                if len(self.replay_buffer) >= self.batch_size:
                    obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.replay_buffer.sample(self.batch_size)

                    # Update the Q network
                    update_dqn(
                        self.q,
                        self.q_target,
                        self.optimizer,
                        self.gamma,
                        num_steps,
                        obs_batch.float(),
                        act_batch,
                        rew_batch.float(),
                        next_obs_batch.float(),
                        tm_batch.float()
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
