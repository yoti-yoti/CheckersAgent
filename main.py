import argparse
from enum import Enum
from envs.make_env import make_env
from agents.checkers_agent import CheckersAgent
import os
import torch
from dotenv import load_dotenv
import training.train as train_module
import networks
import envs


def initialize(network_type: str, params_path: str, opponent_policy: str, opp_params_path: str):
    """Initialize the network based on the type and parameters."""
    print("Creating Checkers environment with opponent policy:", opponent_policy)

    load_dotenv()

    # Optional: let user force a device in .env
    device_str = os.getenv("DEVICE", "auto").lower()

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS GPU")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        # Use the device explicitly from .env, e.g., "cpu", "cuda:0", "mps"
        device = torch.device(device_str)
        print(f"Using device from .env: {device}")
    
    opp_agent = CheckersAgent(network_name=opponent_policy, device=device, checkpoint_id=opp_params_path)
    env = make_env("Checkers-v0", opponent_policy=opp_agent)
    print(f"Initializing {network_type} network with parameters from {params_path}") 

    agent = CheckersAgent(network_name=network_type, device=device, checkpoint_id=params_path)
    return env, agent


def train(network_type: str, params_path: str, opponent_policy: str, opp_params_path: str, number_of_eps: int = 10,):
    """Train the checkers agent."""
    print(f"Training with {network_type} network")
    print(f"Loading parameters from: {params_path}")
    env, agent = initialize(network_type, params_path, opponent_policy, opp_params_path)
    print(f"Starting training for {number_of_eps} episodes")
    train_module.train(env, agent, number_of_eps)
    agent.save(base_dir="checkpoints", network_name=network_type)
    
    

def play(network_type: str, params_path: str):
    """Play checkers with the trained agent."""
    print(f"Playing with {network_type} network")
    print(f"Loading model from: {params_path}")
    # TODO: Implement play logic
    pass

def main():
    parser = argparse.ArgumentParser(description="Checkers Agent")
    parser.add_argument(
        "mode",
        choices=["train", "play"],
        help="Mode to run: train or play"
    )
    parser.add_argument(
        "--network",
        type=str,
        required=True,
        help="Network type to use"
    )
    parser.add_argument(
        "--opp",
        type=str,
        required=True,
        help="Opponent policy to use"
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to parameters file"
    )
    parser.add_argument(
        "--oppparams",
        type=str,
        default=None,
        help="Path to parameters file for opponent"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training (default: 10)"
    )
    
    args = parser.parse_args()
    
    network_type = args.network
    opponent_policy = args.opp
    
    if args.mode == "train":
        train(network_type, args.params, opponent_policy, args.oppparams, args.epochs)
    elif args.mode == "play":
        play(network_type, args.params)


if __name__ == "__main__":
    main()