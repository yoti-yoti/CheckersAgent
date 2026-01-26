import argparse
import os
import torch
from dotenv import load_dotenv

from envs.make_env import make_env
from envs.checkers_renderer import CheckersRenderer
from agents.checkers_agent import CheckersAgent
from training.trainer import Trainer


def get_device():
    load_dotenv()
    device_str = os.getenv("DEVICE", "auto").lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_agents_and_env(args, device):
    opp_agent = CheckersAgent(
        network_name=args.opp,
        device=device,
        checkpoint_id=args.oppparams,
        player="opponent",
        eps=0.0,
    )

    env = make_env(
        args.env,
        opponent_policy=opp_agent,
        render_mode=None,
        seed=args.seed,
    )

    agent = CheckersAgent(
        network_name=args.network,
        device=device,
        checkpoint_id=args.params,
        player="agent",
        eps=0.0,
    )

    renderer = None
    if args.render:
        renderer = CheckersRenderer(scale=args.scale)

    return env, agent, opp_agent, renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "play"])
    parser.add_argument("--env", type=str, default="Checkers-v0")
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--opp", type=str, required=True)
    parser.add_argument("--params", type=int, default=None)
    parser.add_argument("--oppparams", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--scale", type=int, default=80)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--render_every_steps", type=int, default=1)

    parser.add_argument("--rollout_mode", choices=["episode", "steps"], default="episode")
    parser.add_argument("--rollout_steps", type=int, default=2048)

    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    device = get_device()
    print("device:", device)

    env, agent, opp_agent, renderer = build_agents_and_env(args, device)

    if args.mode == "train":
        trainer = Trainer(
            env=env,
            agent=agent,
            opponent=opp_agent,
            renderer=renderer,
            render_every_steps=args.render_every_steps,
            render_fps=args.fps,
            rollout_mode=args.rollout_mode,
            rollout_steps=args.rollout_steps,
            save_every_episodes=args.save_every,
            checkpoint_dir=args.ckpt_dir,
            network_name=args.network,
            device=device,
        )
        trainer.train(num_episodes=args.episodes, max_steps_per_episode=args.max_steps)

    if args.mode == "play":
        raise NotImplementedError


if __name__ == "__main__":
    main()
