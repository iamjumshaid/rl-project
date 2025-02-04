import argparse
import torch
import gymnasium as gym

from base_agent.dqn_agent import DQNAgent as BaseAgent
from per.dqn_agent import DQNAgent as PERAgent
from utils import make_epsilon_greedy_policy, plot_episode_stats, save_episode_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", choices=["base", "per"], help="Type of agent to use: 'base' or 'per'")
    parser.add_argument("env_name", choices=["Breakout", "Asterix", "SpaceInvaders"], help="Environment name (e.g., 'breakout', 'asterix')")
    args = parser.parse_args()

    env = gym.make(f"MinAtar/{args.env_name}-v1", render_mode="rgb_array")
    
    print(f"Training on {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    NUM_EPISODES = 10_000

    base_hyperparameters = {
        "lr": 0.001, 
        "batch_size": 32,
        "maxlen": 100_000,
        "update_freq": 100,
        "eps_start": 1,
        "eps_end": 0.01,
        "schedule_duration": 15_000,
        "gamma": 0.99,
        "device":  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    if args.agent_type == "base":
        agent = BaseAgent(env, **base_hyperparameters)
        stats = agent.train(NUM_EPISODES)
        save_episode_stats(stats, f"results/{args.env_name}_base.csv")
    else:
        per_hyperparameters = {
            "alpha": 0.5,
            "beta": 0.4
        }
        agent = PERAgent(env, **base_hyperparameters, **per_hyperparameters)
        stats = agent.train(NUM_EPISODES)
        save_episode_stats(stats, f"results/{args.env_name}_per.csv")

if __name__ == "__main__":
    main()
