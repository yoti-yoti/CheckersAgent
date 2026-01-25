import numpy as np

DIR_MAP = [(-1, -1), (-1, 1), (1, -1), (1, 1),
               (-2, -2), (-2, 2), (2, -2), (2, 2)]

def get_legal_moves_mask(board, player, prev_board=None, last_action=None):
    legal_moves = generate_legal_moves(
        board,
        player,
        prev_board=prev_board,
        last_action=last_action
    )
    mask = [0] * 256
    for move in legal_moves:
        mask[move] = 1
    return mask

def get_forced_jumps(board, player, last_action) -> bool:
    if last_action is None:
        return False

    from_sq, dir = divmod(last_action, 8)

    # Not a jump
    if dir < 4:
        return False

    r, c = sq_to_rc(from_sq)
    piece = board[r, c]

    normal_dirs, jump_dirs = get_directions(piece, player)

    for dr, dc in jump_dirs:
        mid_r, mid_c = r + dr // 2, c + dc // 2
        to_r, to_c = r + dr, c + dc

        if (
            0 <= to_r < 8 and 0 <= to_c < 8 and
            board[to_r, to_c] == 0 and
            board[mid_r, mid_c] * player < 0  # eats man OR king
        ):
            return True

    return False


def generate_legal_moves(board, player, prev_board=None, last_action=None):
    legal_moves = []
    jump_moves = []

    # 1. Forced continuation jump
    if last_action is not None and get_forced_jumps(board, player, last_action):
        from_sq, _ = divmod(last_action, 8)
        r, c = sq_to_rc(from_sq)
        piece = board[r, c]

        _, jump_dirs = get_directions(piece, player)

        for i, (dr, dc) in enumerate(jump_dirs):
            mid_r, mid_c = r + (dr // 2), c + (dc // 2)
            to_r, to_c = r + dr, c + dc

            if (
                0 <= to_r < 8 and 0 <= to_c < 8 and
                board[to_r, to_c] == 0 and
                board[mid_r, mid_c] * player < 0
            ):
                legal_moves.append(from_sq * 8 + DIR_MAP.index((dr, dc)))

        return legal_moves

    # 2. Scan whole board
    for r in range(8):
        for c in range(8):
            if board[r, c] * player <= 0:
                continue

            piece = board[r, c]
            from_sq = rc_to_sq(r, c)
            normal_dirs, jump_dirs = get_directions(piece, player)

            # Jumps
            for i, (dr, dc) in enumerate(jump_dirs):
                mid_r, mid_c = r + (dr // 2), c + (dc // 2)
                to_r, to_c = r + dr, c + dc

                if (
                    0 <= to_r < 8 and 0 <= to_c < 8 and
                    board[to_r, to_c] == 0 and
                    board[mid_r, mid_c] * player < 0
                ):
                    jump_moves.append(from_sq * 8 + DIR_MAP.index((dr, dc)))

            # Normal moves
            for i, (dr, dc) in enumerate(normal_dirs):
                to_r, to_c = r + dr, c + dc
                if (
                    0 <= to_r < 8 and 0 <= to_c < 8 and
                    board[to_r, to_c] == 0
                ):
                    legal_moves.append(from_sq * 8 + DIR_MAP.index((dr, dc)))

    # 3. Mandatory capture
    if jump_moves:
        return jump_moves

    return legal_moves



def get_directions(piece, player):
    if abs(piece) == 2:  # king
        normal_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        jump_dirs   = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    else:  # man
        normal_dirs = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        jump_dirs   = [(-2, -2), (-2, 2)] if player == 1 else [(2, -2), (2, 2)]

    return normal_dirs, jump_dirs














# def get_legal_moves_mask(board, player, prev_board=None, last_action=None):
#     legal_moves = generate_legal_moves(board, player, prev_board=prev_board, last_action=last_action)
#     mask = [0] * 256  # Assuming 32 squares * 8 directions
#     for move in legal_moves:
#         mask[move] = 1
#     return mask

# def generate_legal_moves(board, player, prev_board=None, last_action=None):
#     legal_moves = []
#     #TODO TODO TODO TODO
#     # Implement logic to generate all legal moves for the current player
#     # This is a placeholder implementation
#     forced_from = last_action if board == board_after_move(prev_board, last_action, player=player) else None

#     if forced_from is not None:
#         # Only consider moves from the forced_from square
#         from_sq, dir = divmod(forced_from, 8)
#         from_r, from_c = divmod(from_sq, 4)
        

#         directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
#         for dr, dc in directions:
#             to_r, to_c = from_r + dr, from_c + dc
#             if 0 <= to_r < 8 and 0 <= to_c < 8 and board[to_r, to_c] == 0:
#                 dir_index = directions.index((dr, dc))
#                 action = from_sq * 8 + dir_index
#                 legal_moves.append(action)
#         return legal_moves
#     for r in range(8):
#         for c in range(8):
#             if board[r, c] * player > 0:  # If it's the player's piece
#                 # Check possible moves (this is simplified)
#                 directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
#                 for dr, dc in directions:
#                     to_r, to_c = r + dr, c + dc
#                     if 0 <= to_r < 8 and 0 <= to_c < 8 and board[to_r, to_c] == 0:
#                         from_sq = (r * 4) + (c // 2)
#                         dir_index = directions.index((dr, dc))
#                         action = from_sq * 8 + dir_index
#                         legal_moves.append(action)
#     return legal_moves

def board_after_move(board, action, player): # TODO
    if board is None or action is None:
        return None, None
    new_board = board.copy()
    from_sq, dir = divmod(action, 8)
    from_r, from_c = divmod(from_sq, 4)
    if from_r % 2 == 0:
        from_c = from_c * 2 + 1
    else:
        from_c = from_c * 2

    # dir_map = [(-1, -1), (-1, 1), (1, -1), (1, 1),
    #            (-2, -2), (-2, 2), (2, -2), (2, 2)]
    
    dr, dc = DIR_MAP[dir]
    to_r, to_c = from_r + dr, from_c + dc
    # Apply move
    if dir >= 4:  # Jump move
        mid_r, mid_c = (from_r + to_r) // 2, (from_c + to_c) // 2
        new_board[mid_r, mid_c] = 0  # Remove jumped piece
    new_board[to_r, to_c] = new_board[from_r, from_c]
    new_board[from_r, from_c] = 0
    if to_r == 0 and player == 1:
        new_board[to_r, to_c] = 2  # Promote to king
    if to_r == 7 and player == -1:
        new_board[to_r, to_c] = -2  # Promote to king

    return new_board


def check_game_status(board, player):
    """
    Given an 8x8 checkers board, returns:
    'win' if player 1 wins,
    'loss' if player 1 loses,
    'draw' if neither can move or no pieces.
    """
    board = np.array(board)
    
    # Count pieces
    player1_pieces = np.sum((board == 1) | (board == 2))
    player2_pieces = np.sum((board == -1) | (board == -2))
    
    if player1_pieces == 0:
        print("Player 1 has no pieces left.")
        return 'loss'
    if player2_pieces == 0:
        print("Player 2 has no pieces left.")
        return 'win'
    
    # Simple legal move check (can be improved with full checkers rules)
    def has_moves(player):
        directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        if player == 1:
            piece_vals = [1, 2]
        else:
            piece_vals = [-1, -2]
        
        for r in range(8):
            for c in range(8):
                if board[r, c] in piece_vals:
                    # Normal moves
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == 0:
                            return True
                    # King moves
                    if abs(board[r, c]) == 2:
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == 0:
                                return True
        return False
    
    player1_can_move = has_moves(1)
    player2_can_move = has_moves(-1)
    
    if not player1_can_move and not player2_can_move:
        print("Neither player can move, draw.")
        return 'draw'
    if not player1_can_move and player == 1:
        print("Player 1 cannot move, loses.")
        return 'loss'
    if not player2_can_move and player == 1:
        print("Player 2 cannot move, Player 1 wins.")
        return 'win'
    
    return 'ongoing'


# def get_forced_jumps(board, player, last_move) -> bool:
#     from_sq, dir = divmod(last_move, 8)
#     from_r, from_c = divmod(from_sq, 4)
#     if from_r % 2 == 0:
#         from_c = from_c * 2 + 1
#     else:
#         from_c = from_c * 2
#     if dir < 4:  # Jump move
#         return False  # No forced jump if last move wasn't a jump
#     else:
#         if dir == 4:  # Forward left
#             to_c = from_c - 2
#         elif dir == 5:  # Forward right
#             to_c = from_c + 2
#         elif dir == 6:  # Backward left
#             to_c = from_c - 2
#         else:  # Backward right
#             to_c = from_c + 2

#     # if found: player will get "another turn" where his only legal moves are the continued jumps
#     return forced_jump

# def check_jump_series(board, from_r, from_c, player, visited=None):
    # if visited is None:
    #     visited = set()
    # jump_series = []
    # #TODO TODO TODO TODO
    # # Implement logic to check for jump series from a given position
    # return jump_series

# def get_forced_jumps(board, player, last_action) -> bool:
#     if last_action is None:
#         return False

#     from_sq, dir = divmod(last_action, 8)

#     # Last move was NOT a jump
#     if dir < 4:
#         return False

#     r, c = sq_to_rc(from_sq)

#     # Jump directions
#     directions = [(-2, -2), (-2, 2)] if player == 1 else [(2, -2), (2, 2)]

#     for dr, dc in directions:
#         mid_r, mid_c = r + dr // 2, c + dc // 2
#         to_r, to_c = r + dr, c + dc

#         if (
#             0 <= to_r < 8 and 0 <= to_c < 8 and
#             board[to_r, to_c] == 0 and
#             board[mid_r, mid_c] == -player
#         ):
#             return True

#     return False

# def generate_legal_moves(board, player, last_action=None):
#     legal_moves = []
#     jump_moves = []

#     normal_dirs = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
#     jump_dirs = [(-2, -2), (-2, 2)] if player == 1 else [(2, -2), (2, 2)]

#     # 1. Forced continuation jump
#     if get_forced_jumps(board, player, last_action):
#         from_sq, _ = divmod(last_action, 8)
#         r, c = sq_to_rc(from_sq)

#         for i, (dr, dc) in enumerate(jump_dirs):
#             mid_r, mid_c = r + dr // 2, c + dc // 2
#             to_r, to_c = r + dr, c + dc

#             if (
#                 0 <= to_r < 8 and 0 <= to_c < 8 and
#                 board[to_r, to_c] == 0 and
#                 board[mid_r, mid_c] == -player
#             ):
#                 action = from_sq * 8 + (4 + i)
#                 legal_moves.append(action)

#         return legal_moves

#     # 2. Scan entire board
#     for r in range(8):
#         for c in range(8):
#             if board[r, c] != player:
#                 continue

#             from_sq = rc_to_sq(r, c)

#             # Check jumps
#             for i, (dr, dc) in enumerate(jump_dirs):
#                 mid_r, mid_c = r + dr // 2, c + dc // 2
#                 to_r, to_c = r + dr, c + dc

#                 if (
#                     0 <= to_r < 8 and 0 <= to_c < 8 and
#                     board[to_r, to_c] == 0 and
#                     board[mid_r, mid_c] == -player
#                 ):
#                     jump_moves.append(from_sq * 8 + (4 + i))

#             # Check normal moves
#             for i, (dr, dc) in enumerate(normal_dirs):
#                 to_r, to_c = r + dr, c + dc
#                 if 0 <= to_r < 8 and 0 <= to_c < 8 and board[to_r, to_c] == 0:
#                     legal_moves.append(from_sq * 8 + i)

#     # 3. If any jump exists, jumps are mandatory
#     if jump_moves:
#         return jump_moves

#     return legal_moves


def sq_to_rc(sq):
    r, c = divmod(sq, 4)
    if r % 2 == 0:
        c = c * 2 + 1
    else:
        c = c * 2
    return r, c


def rc_to_sq(r, c):
    return r * 4 + (c // 2)
