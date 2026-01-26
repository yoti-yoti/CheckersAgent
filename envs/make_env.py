# envs/make_env.py
import gymnasium as gym

def make_env(env_name, opponent_policy=None, render_mode=None, seed=None):
    env = gym.make(env_name, opponent_policy=opponent_policy, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
