import gymnasium as gym
from gymnasium import spaces
import numpy as np
from checkers_moves import check_game_status, board_after_move, get_forced_jumps
from agents.base_agent import BaseAgent


class CheckersEnv(gym.Env):
    """
    Board encoding:
    -2 : opponent king
    -1 : opponent piece
     0 : empty
     1 : my piece
     2 : my king
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy: BaseAgent,render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Observation: 8x8 board
        self.observation_space = spaces.Box(
            low=-2,
            high=2,
            shape=(8, 8),
            dtype=np.int8,
        )

        # from_square (0–31) × move in directions (0–7)
        self.action_space = spaces.Discrete(32 * 8)

        self.board = None
        self.current_player = 1  # 1 = agent, -1 = opponent
        self.opponent_policy_agent = opponent_policy 

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self._create_initial_board()
        self.current_player = 1

        return self.board.copy(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        self.board = board_after_move(self.board, action, self.current_player)
        status = check_game_status(self.board, self.current_player)
        if status == 'win':
            reward = 1.0
            terminated = True
        elif status == 'draw':
            reward = 0.0
            terminated = True
        # TODO:

        # Switch player (for now opponent does nothing)
        if status == 'ongoing' and not get_forced_jumps(self.board, self.current_player, action):
            self.current_player *= -1
            first=True
            opponent_action=None
            while first or get_forced_jumps(self.board, self.current_player, opponent_action):
                first=False
                opponent_action, _, _ = self.opponent_policy_agent.act(self.board)
                # Apply the opponent's action
                self.board = board_after_move(self.board, opponent_action, self.current_player)
            status = check_game_status(self.board, self.current_player)
            if status == 'loss':
                reward = -1.0
                terminated = True
            elif status == 'draw':
                reward = 0.0
                terminated = True
            self.current_player *= -1

        return self.board.copy(), reward, terminated, truncated, info # type: ignore # Should never be None here

    def render(self):
        if self.render_mode == "human":
            print(self.board)

    # -------------------------
    # Helpers
    # -------------------------
    def _create_initial_board(self):
        board = np.zeros((8, 8), dtype=np.int8)

        # Opponent pieces (top)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row, col] = -1

        # Agent pieces (bottom)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row, col] = 1

        return board
    

# Register the environment
gym.register(
    id="Checkers-v0",
    entry_point="envs.checkers_env:CheckersEnv",
)