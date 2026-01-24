# CheckersAgent
Checkers Deep Neural Network agent 

# Overview

This project trains a reinforcement learning agent to play Checkers using Gymnasium and PyTorch, with support for self-play and policy-gradient methods (e.g. PPO-style updates).

The system is designed to be:

Modular – environments, agents, and networks are cleanly separated

Extensible – new networks or environments can be plugged in easily

Gym-compatible – training follows a standard RL loop

High-Level Architecture
+--------------------+
|     train.py       |
|  (training loop)   |
+---------+----------+
          |
          | controls
          v
+--------------------+        +----------------------+
|   CheckersEnv      |<-------|  Opponent Policy     |
| (Gym environment)  |        +----------------------+
+---------+----------+
          ^
          | observations / rewards 
          | moves 
          v
+---------+----------+
|   CheckersAgent    | 
| (rollout + update) |
+---------+----------+
          |
          v
+----------------------+
| Neural Network       |
| (via registry)       |
+----------------------+


Flow per episode:

Environment provides an observation (board state)

Agent selects a legal action using the network

Environment applies the move (and opponent move)

Agent stores rollout data

At episode end, advantages are computed and the network is updated

Model checkpoint is saved after all episodes

# Getting Started
1. Create the Conda Environment

All dependencies are defined in environment.yml.

conda env create -f environment.yml
conda activate draughts


(Using Conda is recommended but not strictly required as long as dependencies match.)

2. Train an Agent
python main.py train --network network1 --params ./network1/checkpoint_10 --epochs 1000


--network selects a model from the network registry

--epochs corresponds to the number of played episodes

Models are automatically saved as checkpoints under:

./<network_name>/checkpoint_<id>/

3. Play Against a Trained Agent
python main.py play --network network1 --params ./network1/checkpoint_10


Loads the specified checkpoint

Runs the environment in play mode

You act as the opponent against the trained policy
(play interface is currently not implemented / TODO)

Extending the Project

Add a new network
Create a new file under networks/, implement forward, and register it: @register_network("<name_of_network>").

Change learning logic
Modify learn_from_rollout in CheckersAgent.

Add a new environment
Implement a Gym-compatible environment and wire it through make_env.

The training loop remains unchanged as long as the agent extends BaseAgent.


# Project TODOS:
- go over all TODOs in code
- ~~implement `learn_from_rollout` TODO 1~~
- implement `step` environment logic TODO 2
- implement legal moves and jump logic in `checkers_moves.py` TODO 3
- implement human player interface TODO 4
- create networks, and run actual self play and training
- add data collection for analysis
- testing

# Implementation Notes
- Board (observation space) implemented as 8x8 matrix, -2 is opp king, -1 is opp piece, 0 is empty square, 1 is agent piece, 2 is agent king.
- action space is definced by 32x8 space (choice between 256 possible moves, not all of which are legal) from square, and direction
- mask should check each move and choose whether it is legal or not, 1 if legal and 0 if illegal
- 
Questions:
- Should forced moves be made automatically or should they just be masked?
- 

# Files
- `base_agent.py` : abstract class base for agent
- `checkers_agent.py` : checkers agent which extends `BaseAgent`, it is initialized with the network type registered in the network registry, and a checkpoint of the network which it then loads from the networks saved, and the device to store the network. Methods:
    - `act` : recieves an observation (current board state), puts it through the network, applies the legal moves mask, and returns the state(currently argmax, could be changed to sample), its probability, and the value.
    - `update` : after a move, stores the information for the rollout
    - `finish_rollout` : calculates the advantages retrospectively and stores in rollout
    - `learn_from_rollout` : TODO 1 - still needs to be implemented, should calculate the loss, and optimize. Need to decide if this is network dependent and make it a network method, or agent method, need to understand exactly how it looks.
    - `save` : saves current network as a new checkpoint in folder `./<network_name>/checkpoint_<id>` (network_name from registry)
    - `load` : loads network from network type and checkpoint id
    - `initialize_network` : if first time network type is used then it initializes new network, TODO implement initialization? random?
- `make_env.py` : makes environment from name
- `checkers_env.py` : creates a checkers environment, the observations space is 8x8 matrix defining the board, action space is the 256 possible moves, 32 squares that a piece can be on and 8 moves(4 directions, jump or no jump). An opponent policy must be chosen for the environment. Methods:
    - `step` : TODO 2 - still needs to finish implementation, it changes the environment according to the move done, should check if theres another forced jump move for the agent, should conatin all the environment change logic and then finally should apply the opponent's move as part of the environment and thend swithcing back to agent's turn.
    - has `render` for showing the board and `reset` for restarting a game.
- `networks/registry.py` : file for registering all of the networks.
- `networks/__init__.py` : file that is imported so that all networks are registered
- `networks/network1.py` : first network example, all networks must have input size 64, the board (should be 8x8), and `forward` that returns `logits` a 256 long vector of the probabilities of all fo the moves, and `value` the network's estimation of the value of the inputted state. TODO implement base_network (abstract?) so all networks are extensions?
- `checkers_moves.py` : TODO 3 - needs to be implemented helper file for computing the legal moves and jumps, ~~perhaps? should be moved to `checkers_env.py`~~
- `train.py` : contains logic for training loop: recieves env, agent, num_of_episodes, and runs loop of a game until it is finished, updating the rollout, then applying finish rollout and learn from rollout. Same for any env and agent that are gym environments and extend `BaseAgent`
- `main.py` : main function for training, requires arguments for mode (train or play), --network (agent network to use form registry), --params (path to parameter file to load from(checkpoint)) and --epochs, number of epochs(episodes?) to run
    - if `train` runs training with the inputted network, if `play` TODO 4 - should implement interface for playing against the trained agent that was loaded.