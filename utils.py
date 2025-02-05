# Imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_epsilon_greedy_policy(Q: nn.Module, num_actions: int):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon. Taken from last exercise with changes.

    :param Q: The DQN network.
    :param num_actions: Number of actions in the environment.

    :returns: A function that takes the observation as an argument and returns the greedy action in form of an int.
    """

    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        """This function takes in the observation and returns an action."""
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)
        
        # For action selection, we do not need a gradient and so we call ".detach()"
        return Q(obs).argmax().detach()

    return policy_fn


def linear_epsilon_decay(eps_start: float, eps_end: float, current_timestep: int, duration: int) -> float:
    """
    Linear decay of epsilon.

    :param eps_start: The initial epsilon value.
    :param eps_end: The final epsilon value.
    :param current_timestep: The current timestep.
    :param duration: The duration of the schedule (in timesteps). So when schedule_duration == current_timestep, eps_end should be reached

    :returns: The current epsilon.
    """

    if current_timestep > duration:
        return eps_end
    
    fraction_remaining = (duration - current_timestep) / duration

    return eps_end + (eps_start - eps_end) * fraction_remaining

def plot_episode_stats(stats, smoothing_window=500):
    """
    Plots the episode length and smoothed episode reward over time.
    
    Parameters:
        stats: An object with 'episode_lengths' and 'episode_rewards' attributes.
        smoothing_window: The window size for smoothing the rewards.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    
    # Plot the episode length over time
    ax = axes[0]
    ax.plot(stats.episode_lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length over Time") 
    
    # Plot the episode reward over time
    ax = axes[1]
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")
    
    plt.show()
    
def save_episode_stats(stats, save_path):
    """
    Saves episode statistics to a CSV file.
    
    Parameters:
        stats: An object with 'episode_lengths' and 'episode_rewards' attributes.
        save_path: Path to save the CSV file.
    """
    df = pd.DataFrame({
        "episode": range(1, len(stats.episode_lengths) + 1),
        "length": stats.episode_lengths,
        "reward": stats.episode_rewards
    })
    df.to_csv(save_path, index=False)
    

def plot_comparison_stats(based_stats, per_stats, smoothing_window=500, env_name="", title1="", title2=""):
    """
    Plots the smoothed episode rewards for two different sets of statistics.
    
    Parameters:
        based_stats: Stats object with 'episode_rewards' attribute (baseline stats).
        per_stats: Stats object with 'episode_rewards' attribute (performance stats).
        smoothing_window: The window size for smoothing the rewards.
    """
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    
    rewards_smoothed_based = based_stats["reward"].rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_per = per_stats["reward"].rolling(smoothing_window, min_periods=smoothing_window).mean()
    
    ax.plot(rewards_smoothed_based, label=f"{title1}", color="blue")
    ax.plot(rewards_smoothed_per, label=f"{title2}", color="red")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title(f"Comparison of Episode Rewards Over Time (Smoothed over window size {smoothing_window})")
    ax.legend()
    
    plt.show()