import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import itertools

from utils import linear_epsilon_decay, make_epsilon_greedy_policy
from DQN import DQN
from replay_buffer import ReplayBuffer
import torch.nn.functional as F

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
    steps: torch.Tensor,
    indices: list,
    weights: torch.Tensor,
    memory: ReplayBuffer,
    priority_eps: float = 1e-6,
):
    """
    Update the DQN network for one optimizer step using multi-step targets (or one-step if num_steps = 1).

    :param q: The DQN network.
    :param q_target: The target DQN network.
    :param optimizer: The optimizer.
    :param gamma: The discount factor.
    :param obs: Batch of current observations.
    :param act: Batch of actions.
    :param rew: Batch of multi-step returns.
    :param next_obs: Batch of next observations (from the last step in the multi-step sequence).
    :param tm: Batch of termination flags.
    :param steps: Batch of actual multi-step lengths used per sample.
    """

    with torch.no_grad():
        # Compute discount factors: gamma^(actual_steps)
        discount_factors = torch.pow(
            torch.tensor(gamma, device=rew.device), steps.float()
        )
        action_selection = torch.argmax(q(next_obs), dim=1)
        q_next_eval = (
            q_target(next_obs)
            .gather(dim=1, index=action_selection.unsqueeze(1))
            .squeeze(1)
        )
        td_target = rew + discount_factors * q_next_eval * (1 - tm.float())

    predicted_q = torch.gather(q(obs), dim=1, index=act.unsqueeze(1)).squeeze(1)
    elementwise_loss = F.smooth_l1_loss(predicted_q, td_target, reduction="none")

    loss = torch.mean(elementwise_loss * weights)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    td_delta = elementwise_loss.detach().cpu().numpy()
    new_priorities = td_delta + priority_eps
    memory.update_experience_priorities(indices, new_priorities)

    return loss


class DQNAgent:
    def __init__(
        self,
        env,
        gamma=0.99,
        num_steps=3,  # If num_steps = 1 then base DQN will run
        lr=0.001,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.1,
        schedule_duration=10_000,
        update_freq=100,
        maxlen=100_000,
        alpha=0.5,
        beta=0.4,
    ):
        """
        Initialize the DQN agent.
        """
        self.env = env
        self.gamma = gamma
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_duration = schedule_duration
        self.update_freq = update_freq
        self.alpha = alpha
        self.beta = beta

        self.replay_buffer = ReplayBuffer(maxlen, num_steps, gamma, alpha)

        self.q = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q_target = DQN(self.env.observation_space.shape, self.env.action_space.n)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)

    def train(self, num_episodes: int, n_steps: int):
        """
        Train the DQN agent.

        :param num_episodes: Number of episodes to train.
        :returns: A tuple (stats, loss_history, best_model_state) where:
                  - stats is a namedtuple with episode_lengths and episode_rewards,
                  - loss_history is a list of loss values from training updates,
                  - best_model_state is the state_dict of the best-performing model.
        """
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
            if (i_episode + 1) % 100 == 0:
                print(
                    f"Episode {i_episode + 1} / {num_episodes}  Time Step: {current_timestep}  Epsilon: {epsilon:.3f}"
                )
            obs, _ = self.env.reset()

            episode_reward = 0
            episode_length = 0

            update_count = 0
            loss_sum = 0

            for episode_time in itertools.count():
                epsilon = linear_epsilon_decay(
                    self.eps_start,
                    self.eps_end,
                    current_timestep,
                    self.schedule_duration,
                )
                action = self.policy(
                    torch.as_tensor(obs).unsqueeze(0).float(), epsilon=epsilon
                )
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1

                self.replay_buffer.store(
                    torch.as_tensor(obs),
                    torch.as_tensor(action),
                    torch.as_tensor(reward),
                    torch.as_tensor(next_obs),
                    torch.as_tensor(terminated),
                )

                # Linearly increase beta
                fraction = min(i_episode / num_episodes, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                if len(self.replay_buffer) >= self.batch_size:
                    (
                        obs_batch,
                        act_batch,
                        rew_batch,
                        next_obs_batch,
                        tm_batch,
                        steps_batch,
                        indices,
                        weights,
                    ) = self.replay_buffer.sample(self.batch_size, self.beta)

                    loss = update_dqn(
                        self.q,
                        self.q_target,
                        self.optimizer,
                        self.gamma,
                        obs_batch.float(),
                        act_batch,
                        rew_batch.float(),
                        next_obs_batch.float(),
                        tm_batch.float(),
                        steps_batch,
                        indices,
                        weights,
                        self.replay_buffer,
                    )
                    loss_history.append(loss.item())
                    update_count += 1
                    loss_sum += loss.item()

                if current_timestep % self.update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())
                current_timestep += 1

                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs

            stats.episode_rewards[i_episode] = episode_reward
            stats.episode_lengths[i_episode] = episode_length

            episode_update_counts.append(update_count)
            avg_loss = loss_sum / update_count if update_count > 0 else 0
            episode_avg_losses.append(avg_loss)

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_state = self.q.state_dict()

        return (
            stats,
            loss_history,
            best_model_state,
            episode_update_counts,
            episode_avg_losses,
        )
