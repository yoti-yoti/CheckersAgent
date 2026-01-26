import time
import numpy as np
import torch
from datetime import datetime
import os
import pickle
from stats_visualize import visualize


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

        self.stats = {
            "episodes": 0,
            "total_steps": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "episode_returns": [],
            "episode_lengths": [],
            "win_history": [],  # 1=win, 0=draw, -1=loss
            "loss_history": [],
            "start_time": datetime.now().isoformat(),
            "env": str(env),
            "network_name": network_name,
            "rollout_mode": rollout_mode,
            "rollout_steps": rollout_steps,
            "device": str(device),
        }

    def _render(self, obs, info):
        if self.renderer is None:
            return
        self.renderer.draw(obs, info)
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
                + (f" loss={loss['loss']:.4f}" if loss is not None else "")
            )

            if self.save_every_episodes > 0 and (ep % self.save_every_episodes == 0):
                self.agent.save(base_dir=self.checkpoint_dir, network_name=self.network_name)

            # Episode outcome
            if ep_return > 0:
                self.stats["wins"] += 1
                self.stats["win_history"].append(1)
            elif ep_return < 0:
                self.stats["losses"] += 1
                self.stats["win_history"].append(-1)
            else:
                self.stats["draws"] += 1
                self.stats["win_history"].append(0)

            self.stats["episodes"] += 1
            self.stats["total_steps"] = self.global_step
            self.stats["episode_returns"].append(ep_return)
            self.stats["episode_lengths"].append(ep_len)

            if loss is not None:
                self.stats["loss_history"].append(loss)


        self.agent.save(base_dir=self.checkpoint_dir, network_name=self.network_name)
        self.save_stats()
        visualize()
        return episode_returns, episode_lengths
    
    def play(self, num_episodes=100, max_steps_per_episode=500, render=False):
        episode_returns = []
        episode_lengths = []

        play_stats = {
            "episodes": 0,
            "total_steps": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "episode_returns": [],
            "episode_lengths": [],
            "win_history": [],
            "start_time": datetime.now().isoformat(),
            "mode": "play",
        }

        global_step = 0

        for ep in range(1, int(num_episodes) + 1):
            obs, info = self.env.reset()
            done = False
            ep_return = 0.0
            ep_len = 0

            while not done:
                mask = self._get_mask(info)

                # IMPORTANT: no gradients, no learning
                with torch.no_grad():
                    action, _, _ = self.agent.act(obs, mask=mask)

                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                done = bool(terminated or truncated)

                ep_return += float(reward)
                ep_len += 1
                global_step += 1

                if render and self.renderer is not None:
                    self._render(next_obs, next_info)

                obs, info = next_obs, next_info

                if ep_len >= int(max_steps_per_episode):
                    break

            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

            avg_r = float(np.mean(episode_returns[-50:]))
            avg_len = float(np.mean(episode_lengths[-50:]))

            print(
                f"[PLAY] ep={ep} return={ep_return:.3f} len={ep_len} "
                f"avg50_return={avg_r:.3f} avg50_len={avg_len:.1f}"
            )

            # Outcome stats
            if ep_return > 0:
                play_stats["wins"] += 1
                play_stats["win_history"].append(1)
            elif ep_return < 0:
                play_stats["losses"] += 1
                play_stats["win_history"].append(-1)
            else:
                play_stats["draws"] += 1
                play_stats["win_history"].append(0)

            play_stats["episodes"] += 1
            play_stats["total_steps"] = global_step
            play_stats["episode_returns"].append(ep_return)
            play_stats["episode_lengths"].append(ep_len)

        play_stats["end_time"] = datetime.now().isoformat()
        self.stats = play_stats  # Save for potential later use
        self.save_stats()
        visualize()
        return episode_returns, episode_lengths, play_stats

    
    def save_stats(self):
        # Base directory: training/training_stats
        base_dir = os.path.join("training", "training_stats")
        os.makedirs(base_dir, exist_ok=True)

        # Timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_{self.network_name}_{timestamp}.pkl"
        path = os.path.join(base_dir, filename)

        # Add end time metadata
        self.stats["end_time"] = datetime.now().isoformat()

        with open(path, "wb") as f:
            pickle.dump(self.stats, f)

        print(f"[Trainer] Stats saved to {path}")

