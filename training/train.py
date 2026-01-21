# training/train.py
def train(env, agent, num_episodes):
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update((obs, action, reward, next_obs, done))
            obs = next_obs
            ep_reward += reward

        print(f"Episode {ep}, reward: {ep_reward}")
