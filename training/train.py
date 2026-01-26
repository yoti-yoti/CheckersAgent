import time
import numpy as np
import torch


class Trainer:
    def __init__(
        self,
        env,
        agent,
        opponent=None,
        renderer=None,
        render_every_steps=1,
        render_fps=10,
        rollout_mode="episode",
        rollout_steps=2048,
        save_every_episodes=50,
        checkpoint_dir="checkpoints",
        network_name="checkers_network1",
        device=torch.device("cpu"),
    ):
        self.env = env
        self.agent = agent
        self.opponent = opponent
        self.renderer = renderer
        self.render_every_steps = int(render_every_steps)
        self.render_dt = 1.0 / float(render_fps) if render_fps else 0.0
        self.rollout_mode = rollout_mode
        self.rollout_steps = int(rollout_steps)
        self.save_every_episodes = int(save_every_episodes)
        self.checkpoint_dir = checkpoint_dir
        self.network_name = network_name
        self.device = device

        self.global_step = 0

    def _render(self, obs, info):
        if self.renderer is None:
            return
        self.renderer.render(obs, info)
        if self.render_dt > 0:
            time.sleep(self.render_dt)

    def _get_mask(self, info):
        if info is None:
            return None
        return info.get("action_mask", None)

    def train(self, num_episodes=1000, max_steps_per_episode=500):
        episode_returns = []
        episode_lengths = []

        for ep in range(1, int(num_episodes) + 1):
            obs, info = self.env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                mask = self._get_mask(info)
                action, log_prob, value = self.agent.act(obs, mask=mask)

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = bool(terminated or truncated)

                transition = {
                    "obs": np.asarray(obs, dtype=np.float32),
                    "actions": int(action),
                    "log_probs": log_prob.detach(),
                    "rewards": float(reward),
                    "next_obs": np.asarray(next_obs, dtype=np.float32),
                    "dones": float(done),
                    "values": value.detach(),
                    "masks": np.asarray(mask, dtype=np.int8) if mask is not None else np.zeros(256, dtype=np.int8),
                }
                self.agent.update(transition)

                ep_return += float(reward)
                ep_len += 1
                self.global_step += 1

                if self.renderer is not None and (self.global_step % self.render_every_steps == 0):
                    self._render(next_obs, next_info)

                obs, info = next_obs, next_info

                if self.rollout_mode == "steps":
                    if len(self.agent.update_data["obs"]) >= self.rollout_steps:
                        rollout = self.agent.finish_rollout()
                        loss = self.agent.learn_from_rollout(rollout)
                        break

                if ep_len >= int(max_steps_per_episode):
                    break

            if self.rollout_mode == "episode":
                rollout = self.agent.finish_rollout()
                loss = self.agent.learn_from_rollout(rollout)
            else:
                loss = None

            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

            avg_r = float(np.mean(episode_returns[-50:])) if len(episode_returns) >= 1 else ep_return
            avg_len = float(np.mean(episode_lengths[-50:])) if len(episode_lengths) >= 1 else ep_len

            print(
                f"ep={ep} return={ep_return:.3f} len={ep_len} avg50_return={avg_r:.3f} avg50_len={avg_len:.1f}"
                + (f" loss={loss:.4f}" if loss is not None else "")
            )

            if self.save_every_episodes > 0 and (ep % self.save_every_episodes == 0):
                self.agent.save(base_dir=self.checkpoint_dir, network_name=self.network_name)

        self.agent.save(base_dir=self.checkpoint_dir, network_name=self.network_name)
        return episode_returns, episode_lengths
