import gymnasium as gym

def make_env(env_name, seed=None):
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
    return env

    

