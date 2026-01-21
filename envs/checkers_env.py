import gymnasium as gym
from gymnasium import spaces
import numpy as np


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

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Observation: 8x8 board
        self.observation_space = spaces.Box(
            low=-2,
            high=2,
            shape=(8, 8),
            dtype=np.int8,
        )

        # Action space (TEMPORARY):
        # from_square (0–31) × move in directions (0–7)
        self.action_space = spaces.Discrete(32 * 8)

        self.board = None
        self.current_player = 1  # 1 = agent, -1 = opponent
        self.opponent_policy = None  # Placeholder for opponent policy

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = self._create_initial_board()
        self.current_player = 1

        return self.board.copy(), {}

    def step(self, action):
        from_sq, dir = divmod(action, 8)
        from_r, from_c = divmod(from_sq, 4)
        if from_r % 2 == 0:
            from_c = from_c * 2 + 1
        else:
            from_c = from_c * 2

        # TODO Make dir_map global?
        dir_map = [(-1, -1), (-1, 1), (1, -1), (1, 1),
                   (-2, -2), (-2, 2), (2, -2), (2, 2)]
        dr, dc = dir_map[dir]
        to_r, to_c = from_r + dr, from_c + dc

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # VERY basic legality check (placeholder)
        piece = self.board[from_r, from_c]
        if piece * self.current_player <= 0:
            reward = -1.0  # illegal move
            terminated = True
            return self.board.copy(), reward, terminated, truncated, info

        # Apply move (no capture logic yet)
        self.board[to_r, to_c] = self.board[from_r, from_c]
        self.board[from_r, from_c] = 0

        # TODO:
        # - capture logic
        # - king promotion
        # - forced jumps
        # - win/loss detection

        # Switch player (for now opponent does nothing)
        self.current_player *= -1

        return self.board.copy(), reward, terminated, truncated, info

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