import os
import datetime
import argparse
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np

from IPython.display import Image as IImage
from PIL import Image

from dqn_agent import DQNAgent
from utils import make_epsilon_greedy_policy

def save_rgb_animation(rgb_arrays, filename, duration=50):
    """Save an animated GIF from a list of RGB arrays."""
    frames = []
    for rgb_array in rgb_arrays:
        # Scale to [0,255] and convert to uint8
        rgb_array = (rgb_array * 255).astype(np.uint8)
        # Upscale (repeat each pixel) for visualization (adjust as needed)
        rgb_array = rgb_array.repeat(48, axis=0).repeat(48, axis=1)
        img = Image.fromarray(rgb_array)
        frames.append(img)
    # Save as animated GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def rendered_rollout(policy, env, max_steps=1_000):
    """Rollout for one episode while saving all rendered images."""
    obs, _ = env.reset()
    imgs = [env.render()]
    for _ in range(max_steps):
        action = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        if terminated or truncated:
            break
    return imgs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multistep', action='store_true',
                        help='Enable multi-step learning (if not set, runs base DQN with one-step updates)')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Number of steps for multi-step learning (used only if --multistep is enabled)')
    parser.add_argument('--env', type=str, default='MinAtar/Breakout-v1',
                        help='Name of the environment to use')
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="rgb_array")
    print(f"Training on {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    # Hyperparameters
    LR = 0.001
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 100_000
    UPDATE_FREQ = 100
    EPS_START = 1.0
    EPS_END = 0.01
    SCHEDULE_DURATION = 15_000
    NUM_EPISODES = 10_000
    DISCOUNT_FACTOR = 0.99

    # Decide on the number of steps based on the flag:
    if args.multistep:
        N_STEPS = args.n_steps
        run_type = "multistep"
        print(f"Running with multi-step learning (N_STEPS = {N_STEPS})")
    else:
        N_STEPS = 1
        run_type = "base"
        print("Running with base DQN (one-step learning)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_folder = os.path.join("results", timestamp)
    os.makedirs(timestamp_folder, exist_ok=True)

    run_folder_name = "multistep_dqn_results" if run_type == "multistep" else "base_dqn_results"
    run_folder = os.path.join(timestamp_folder, run_folder_name)
    os.makedirs(run_folder, exist_ok=True)

    # Save hyperparameters into info.txt
    hyperparams = {
        "ENVIRONMENT": args.env,
        "LR": LR,
        "BATCH_SIZE": BATCH_SIZE,
        "REPLAY_BUFFER_SIZE": REPLAY_BUFFER_SIZE,
        "UPDATE_FREQ": UPDATE_FREQ,
        "EPS_START": EPS_START,
        "EPS_END": EPS_END,
        "SCHEDULE_DURATION": SCHEDULE_DURATION,
        "NUM_EPISODES": NUM_EPISODES,
        "DISCOUNT_FACTOR": DISCOUNT_FACTOR,
        "N_STEPS": N_STEPS,
        "MULTISTEP": args.multistep
    }
    info_path = os.path.join(timestamp_folder, "info.txt")
    with open(info_path, "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    print(f"Hyperparameters saved to {info_path}")

    # Train the Agent
    training_start_time = datetime.datetime.now()
    agent = DQNAgent(
        env,
        gamma=DISCOUNT_FACTOR,
        num_steps=N_STEPS,
        lr=LR,
        batch_size=BATCH_SIZE,
        eps_start=EPS_START,
        eps_end=EPS_END,
        schedule_duration=SCHEDULE_DURATION,
        update_freq=UPDATE_FREQ,
        maxlen=REPLAY_BUFFER_SIZE,
    )
    stats, loss_history, best_model_state, episode_update_counts, episode_avg_losses = agent.train(NUM_EPISODES)

    training_end_time = datetime.datetime.now()
    duration = training_end_time - training_start_time
    with open(info_path, "a") as f:
        f.write(f"Training_Start_Time: {training_start_time}\n")
        f.write(f"Training_End_Time: {training_end_time}\n")
        f.write(f"Training_Duration: {duration}\n")
    print(f"Training time info saved to {info_path}")

    # Save the best model
    best_model_path = os.path.join(run_folder, "best_model.pt")
    torch.save(best_model_state, best_model_path)
    print(f"Best model saved to {best_model_path}")

    results_path = os.path.join(run_folder, "results.txt")
    with open(results_path, "w") as f:
        f.write("Episode\tEpisode_Length\tEpisode_Reward\tTraining_Update_Count\tAverage_Training_Loss\n")
        for i in range(NUM_EPISODES):
            f.write(f"{i+1}\t{stats.episode_lengths[i]}\t{stats.episode_rewards[i]}\t{episode_update_counts[i]}\t{episode_avg_losses[i]}\n")
    print(f"Per-episode results saved to {results_path}")

    smoothing_window = 20

    # Plot: Episode Length and Episode Reward (smoothed)
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes1[0].plot(stats.episode_lengths)
    axes1[0].set_xlabel("Episode")
    axes1[0].set_ylabel("Episode Length")
    axes1[0].set_title("Episode Length over Time")

    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    axes1[1].plot(rewards_smoothed)
    axes1[1].set_xlabel("Episode")
    axes1[1].set_ylabel("Episode Reward (Smoothed)")
    axes1[1].set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")
    graph1_path = os.path.join(run_folder, "episode_stats.png")
    fig1.savefig(graph1_path)
    print(f"Episode stats plot saved to {graph1_path}")

    # Plot: Training Loss over Updates
    if loss_history:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(loss_history)
        ax2.set_xlabel("Training Update")
        ax2.set_ylabel("Loss")
        ax2.set_title("Training Loss over Updates")
        loss_fig_path = os.path.join(run_folder, "training_loss.png")
        fig2.savefig(loss_fig_path)
        print(f"Training loss plot saved to {loss_fig_path}")
    else:
        print("No loss history recorded.")

    # Load the best model
    agent.q.load_state_dict(best_model_state)
    policy = make_epsilon_greedy_policy(agent.q, num_actions=env.action_space.n)
    imgs = rendered_rollout(policy, env)
    gif_path = os.path.join(run_folder, "trained.gif")
    save_rgb_animation(imgs, gif_path, duration=50)
    print(f"Trained rollout animation saved to {gif_path}")

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
