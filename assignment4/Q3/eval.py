import argparse
import importlib
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env


def parse_arguments():
    parser = argparse.ArgumentParser(description="DRL HW4 Q3 - DMC Humanoid Run Environment")
    parser.add_argument("--episodes", default=100, type=int, help="Number of episodes to evaluate")
    parser.add_argument("--record_demo", action="store_true", help="Record a demonstration")
    return parser.parse_args()


def load_agent(agent_path):
    """Dynamically load the student's agent class."""
    spec = importlib.util.spec_from_file_location("student_agent", agent_path)
    student_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_module)
    return student_module.Agent()


def make_env():
    # Create Humanoid Run environment.
    env_name = "humanoid-run"
    env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env


def record_video(env, agent):
    import imageio

    gif_path = "./demo.gif"
    state, info = env.reset()
    frames = []

    while True:
        frame = env.render()
        frames.append(np.array(frame))
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state

        if terminated or truncated:
            break

    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved to {gif_path}")


def eval_score():
    """Evaluate the agent's performance on humanoid-run."""
    args = parse_arguments()

    env = make_env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    agent = load_agent("student_agent.py")

    if args.record_demo:
        record_video(env, agent)

    episode_rewards = []
    for _ in tqdm(range(args.episodes), desc="Evaluating"):
        observation, info = env.reset(seed=np.random.randint(0, 1000000))

        episode_reward = 0.0
        done = False
        while not done:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

    env.close()

    mean = np.mean(episode_rewards)
    std = np.std(episode_rewards)
    final_score = np.round(mean - std, 2)

    print("\nEvaluation complete!")
    print(f"Average return over {args.episodes} episodes: {mean:.2f} (std: {std:.2f})")
    print(f"Final score (Mean - Std): {final_score:.2f}")

    return final_score


if __name__ == "__main__":
    score = eval_score()

    # Q3: only calculate the first 15% component: Score / 1000
    local_grade_max = 15.0

    local_grade = np.clip(score / 1000.0, 0.0, 1.0) * local_grade_max

    print("\nQ3 Grading Result (first 15% only)")
    print(f"Student earned: {local_grade:.2f}% / {local_grade_max:.0f}%")
