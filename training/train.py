import numpy as np

def train(env, agent, num_episodes):
    total_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # print(obs)
            
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.update((obs, action, log_prob, reward, next_obs, done, value))
            obs = next_obs
            ep_reward += reward
        total_rewards.append(ep_reward)
        rollout = agent.finish_rollout()
        agent.learn_from_rollout(rollout)

        print(f"Episode {ep}, reward: {ep_reward}")
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards)}")
