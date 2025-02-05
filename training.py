import argparse
import torch
import gymnasium as gym

from base_agent.dqn_agent import DQNAgent as BaseAgent
from final_agent.dqn_agent import DQNAgent as FinalAgent
from integrated_agent.dqn_agent import DQNAgent as IntegratedAgent

from utils import save_episode_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_type", choices=["base", "integrated", "final"], help="Type of agent to use: 'base' or 'per'")
    parser.add_argument("env_name", choices=["Breakout", "Asterix", "SpaceInvaders"], help="Environment name (e.g., 'breakout', 'asterix')")
    args = parser.parse_args()

    # Choose your environment
    env = gym.make("MinAtar/SpaceInvaders-v1", render_mode="rgb_array")

    # Print observation and action space infos
    print(f"Training on {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    NUM_EPISODES = 5_000
    AVG_FRAME_PER_EPISODE = 200

    # Hyperparameters, Hint: Change as you see fit
    base_hyperparameters = {
            "lr": 0.00025, 
            "batch_size": 32,
            "maxlen": 100_000,
            "eps_start": 1,
            "eps_end": 0.01,
            "gamma": 0.99,
            "update_freq": NUM_EPISODES / 25,
            "schedule_duration": AVG_FRAME_PER_EPISODE * NUM_EPISODES * 0.02,
            "training_start": 0,
            "device":  torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
    
    if args.agent_type == "base":
        agent = BaseAgent(env, **base_hyperparameters)
        results = agent.train(NUM_EPISODES)
        save_episode_stats(results[0], f"results/{args.env_name}_base.csv")
    if args.agent_type == "integrated":
        base_hyperparameters["lr"] = base_hyperparameters["lr"] / 4
        integrated_hyperparameters = {
            "alpha": 0.5,
            "beta": 0.4,
            "num_steps": 3,
        }
        agent = IntegratedAgent(env, **base_hyperparameters, **integrated_hyperparameters)
        results = agent.train(NUM_EPISODES)
        save_episode_stats(results[0], f"results/{args.env_name}_integrated_no_ddqn.csv")
    else:
        final_hyperparameters = {
            "num_steps": 5,
        }
        agent = FinalAgent(env, **base_hyperparameters, **final_hyperparameters)
        results = agent.train(NUM_EPISODES)
        
        save_episode_stats(results[0], f"results/{args.env_name}_final_5_steps.csv")
        
if __name__ == "__main__":
    main()
