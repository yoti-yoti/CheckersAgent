import argparse
from enum import Enum

class NetworkType(Enum): #TODO change to be the different networks
    CNN = "cnn"
    DENSE = "dense"
    TRANSFORMER = "transformer"

def train(network_type: NetworkType, params_path: str):
    """Train the checkers agent."""
    print(f"Training with {network_type.value} network")
    print(f"Loading parameters from: {params_path}")
    # TODO: Implement training logic
    pass

def play(network_type: NetworkType, params_path: str):
    """Play checkers with the trained agent."""
    print(f"Playing with {network_type.value} network")
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
        choices=[nt.value for nt in NetworkType],
        required=True,
        help="Network type to use"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to parameters file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training (required if mode is train)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train" and args.epochs is None:
        parser.error("--epochs is required when mode is 'train'")
    
    network_type = NetworkType(args.network)
    
    if args.mode == "train":
        train(network_type, args.params)
    elif args.mode == "play":
        play(network_type, args.params)


if __name__ == "__main__":
    main()