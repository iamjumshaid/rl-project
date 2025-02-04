import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import itertools
from utils import linear_epsilon_decay, make_epsilon_greedy_policy
from DQN import DQN
from base_agent.replay_buffer import ReplayBuffer


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
        q_s_prime = q_target(next_obs)
        max_q_s_prime = torch.max(q_s_prime, dim=1)[0]
        td_target = rew + gamma * max_q_s_prime * ~tm
    # Calculate the loss. Hint: Pytorch has the ".gather()" function, which collects values along a specified axis using some specified indexes
    q_s_a = torch.gather(q(obs), dim=1, index=act.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(q_s_a, td_target)
    # Backpropagate the loss and step the optimizer
    loss.backward()
    optimizer.step()
    return loss


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


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
        training_start=5000,
        device=None,
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
        self.training_start = training_start
        self.device = device

        # Initialize the Replay Buffer
        self.replay_buffer = ReplayBuffer(maxlen)

        # Initialize the Deep Q-Network. Hint: Remember observation_space and action_space
        self.q = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q.to(self.device)

        # Initialize the second Q-Network, u
        # the target network. Load the parameters of the first one into the second
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

        loss_history = []
        best_reward = -float("inf")
        best_model_state = None

        # Record per-episode training update count and average training loss
        episode_update_counts = []
        episode_avg_losses = []

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print(
                    f"Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}  Epsilon: {epsilon:.3f} Training Started: {len(self.replay_buffer) > self.training_start}"
                )
            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            episode_reward = 0
            episode_length = 0

            update_count = 0
            loss_sum = 0

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
                action = self.policy(
                    torch.as_tensor(obs).unsqueeze(0).float(), epsilon=epsilon
                )
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                # Update statistics
                episode_reward += reward
                episode_length += 1
                # Store sample in the replay buffer
                self.replay_buffer.store(
                    torch.as_tensor(obs),
                    torch.as_tensor(action),
                    torch.as_tensor(reward),
                    torch.as_tensor(next_obs),
                    torch.as_tensor(terminated),
                )
                # Sample a mini batch from the replay buffer
                
                # Only train after a certain amount of transitions are present in the buffer
                if len(self.replay_buffer) < self.training_start:
                    continue

                obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = (
                    self.replay_buffer.sample(self.batch_size)
                )

                obs_batch = obs_batch.to(self.device)
                act_batch = act_batch.to(self.device)
                rew_batch = rew_batch.to(self.device)
                next_obs_batch = next_obs_batch.to(self.device)
                tm_batch = tm_batch.to(self.device)

                # Update the Q network
                loss = update_dqn(
                    self.q,
                    self.q_target,
                    self.optimizer,
                    self.gamma,
                    obs_batch.float(),
                    act_batch,
                    rew_batch.float(),
                    next_obs_batch.float(),
                    tm_batch,
                )
                loss_history.append(loss.item())
                update_count += 1
                loss_sum += loss.item()

                # Update the current Q target
                if current_timestep % self.update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())
                current_timestep += 1
                # Check whether the episode is finished
                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs

            stats.episode_rewards[i_episode] = episode_reward
            stats.episode_lengths[i_episode] = episode_length


        return (
            stats,
            loss_history,
            best_model_state,
            episode_update_counts,
            episode_avg_losses,
        )
