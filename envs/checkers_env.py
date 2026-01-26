# envs/checkers_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents.base_agent import BaseAgent
from envs.checkers_renderer import CheckersRenderer
from checkers_moves import (
    check_game_status,
    board_after_move,
    get_forced_jumps,
    generate_legal_moves,
    get_legal_moves_mask,
    DIR_MAP,
    sq_to_rc,
    rc_to_sq,
)

class CheckersEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy: BaseAgent, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-2, high=2, shape=(8, 8), dtype=np.int8)
        self.action_space = spaces.Discrete(32 * 8)

        self.board = None
        self.current_player = 1
        self.opponent_policy_agent = opponent_policy

        self.pending_player = None
        self.pending_last_action = None

        self.render_fps = 30
        self.hud = {}
        self.renderer = None
        if self.render_mode == "human":
            self.renderer = CheckersRenderer(fps=self.render_fps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._create_initial_board()
        self.current_player = 1
        self.pending_player = None
        self.pending_last_action = None

        # Randomly choose starting player
        self.current_player = self.np_random.choice([1, -1])

        # If opponent starts, play their turn now
        if self.current_player == -1:
            opponent_last_action = None

            while True:
                opp_legal = self.legal_actions(self.current_player, opponent_last_action)
                if len(opp_legal) == 0:
                    break

                # 1. Prepare board for opponent
                opponent_board = self._prepare_opponent_board(self.board)

                # 2. Get action from opponent policy
                opp_action_flipped, _, _ = self.opponent_policy_agent.act(opponent_board)

                # 3. Translate to real board coordinates
                opp_action_real = self._translate_opponent_action(opp_action_flipped)

                # 4. Ensure legality
                if opp_action_real not in opp_legal:
                    opp_action_real = int(np.random.choice(opp_legal))

                # 5. Apply move
                self.board = board_after_move(self.board, opp_action_real, self.current_player)

                # if opponent_action not in opp_legal:
                #     opponent_action = int(self.np_random.choice(opp_legal))

                # self.board = board_after_move(self.board, opponent_action, self.current_player)

                if self._must_continue(self.current_player, opp_action_real):
                    opponent_last_action = opp_action_real
                    continue

                break

            self.current_player = 1  # agent always acts after reset

        info = {"action_mask": self.action_mask(self.current_player, None)}
        return self.board.copy(), info

    def step(self, action):
        last_action = self.pending_last_action if self.pending_player == self.current_player else None
        legal = self.legal_actions(self.current_player, last_action)

        if action not in legal:
            info = {"illegal_action": True, "action_mask": self.action_mask(self.current_player, last_action)}
            return self.board.copy(), -1.0, True, False, info

        self.board = board_after_move(self.board, action, self.current_player)

        status = check_game_status(self.board, self.current_player)
        if status == "win":
            info = {"action_mask": self.action_mask(1, None)}
            self.current_player = 1
            return self.board.copy(), 1.0, True, False, info
        if status == "loss":
            info = {"action_mask": self.action_mask(1, None)}
            self.current_player = 1
            return self.board.copy(), -1.0, True, False, info
        if status == "draw":
            info = {"action_mask": self.action_mask(1, None)}
            self.current_player = 1
            return self.board.copy(), 0.0, True, False, info

        if self._must_continue(self.current_player, action):
            self.pending_player = self.current_player
            self.pending_last_action = action
            info = {"must_continue": True, "action_mask": self.action_mask(self.current_player, action)}
            return self.board.copy(), 0.0, False, False, info

        self.pending_player = None
        self.pending_last_action = None

        self.current_player = -1
        opponent_last_action = None

        while True:
            opp_legal = self.legal_actions(self.current_player, opponent_last_action)
            if len(opp_legal) == 0:
                break

            # 1. Prepare board for opponent
            opponent_board = self._prepare_opponent_board(self.board)

            # 2. Get action from opponent policy
            opp_action_flipped, _, _ = self.opponent_policy_agent.act(opponent_board)

            # 3. Translate to real board coordinates
            opp_action_real = self._translate_opponent_action(opp_action_flipped)

            # 4. Ensure legality
            # legal_moves = self.legal_actions(self.current_player, opponent_last_action)
            if opp_action_real not in opp_legal:
                opp_action_real = int(np.random.choice(opp_legal))

            # 5. Apply move
            self.board = board_after_move(self.board, opp_action_real, self.current_player)

            # if opponent_action not in opp_legal:
            #     opponent_action = int(np.random.choice(opp_legal))

            # self.board = board_after_move(self.board, opponent_action, self.current_player)

            status = check_game_status(self.board, self.current_player)

            if status == "win":
                info = {"action_mask": self.action_mask(1, None)}
                self.current_player = 1
                return self.board.copy(), 1.0, True, False, info

            if status == "loss":
                info = {"action_mask": self.action_mask(1, None)}
                self.current_player = 1
                return self.board.copy(), -1.0, True, False, info

            if status == "draw":
                info = {"action_mask": self.action_mask(1, None)}
                self.current_player = 1
                return self.board.copy(), 0.0, True, False, info

            if self._must_continue(self.current_player, opp_action_real):
                opponent_last_action = opp_action_real
                continue

            break

        self.current_player = 1
        info = {"action_mask": self.action_mask(self.current_player, None)}
        return self.board.copy(), 0.0, False, False, info
    
    def _prepare_opponent_board(self, board):
        """Flip board for opponent perspective: 180° + swap signs"""
        board_flipped = np.flip(board, (0, 1)).copy()
        board_flipped = -board_flipped  # Opponent sees their pieces as positive
        return board_flipped

    def _translate_opponent_action(self, action):
        """
        Convert opponent action from their perspective (flipped board) to real board.
        action = from_sq * 8 + dir_index
        """
        from_sq, dir_index = divmod(action, 8)
        r, c = sq_to_rc(from_sq)

        # Flip coordinates 180°
        r, c = 7 - r, 7 - c
        from_sq_real = rc_to_sq(r, c)

        # Flip direction
        dr, dc = DIR_MAP[dir_index]
        dr, dc = -dr, -dc  # invert direction for 180° flip

        # Find the new dir_index in DIR_MAP
        dir_index_real = DIR_MAP.index((dr, dc))

        return from_sq_real * 8 + dir_index_real



    def render(self):
        if self.render_mode != "human":
            return
        if self.renderer is None:
            self.renderer = CheckersRenderer(fps=self.render_fps)
        self.renderer.draw(self.board, self.hud)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def legal_actions(self, player, last_action=None):
        moves = generate_legal_moves(self.board, player, last_action=last_action)
        return [int(a) for a in moves]

    def action_mask(self, player, last_action=None):
        return get_legal_moves_mask(self.board, player, last_action=last_action)

    def _must_continue(self, player, last_action):
        return bool(get_forced_jumps(self.board, player, last_action))

    def _create_initial_board(self):
        board = np.zeros((8, 8), dtype=np.int8)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row, col] = -1
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row, col] = 1
        return board
