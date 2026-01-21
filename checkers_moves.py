# TODO Implement how to generate the mask and legal moves


def get_legal_moves_mask(board, player):
    legal_moves = generate_legal_moves(board, player)
    mask = [0] * 256  # Assuming 32 squares * 8 directions
    for move in legal_moves:
        mask[move] = 1
    return mask

def generate_legal_moves(board, player):
    legal_moves = []
    #TODO TODO TODO TODO
    # Implement logic to generate all legal moves for the current player
    # This is a placeholder implementation
    for r in range(8):
        for c in range(8):
            if board[r, c] * player > 0:  # If it's the player's piece
                # Check possible moves (this is simplified)
                directions = [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
                for dr, dc in directions:
                    to_r, to_c = r + dr, c + dc
                    if 0 <= to_r < 8 and 0 <= to_c < 8 and board[to_r, to_c] == 0:
                        from_sq = (r * 4) + (c // 2)
                        dir_index = directions.index((dr, dc))
                        action = from_sq * 8 + dir_index
                        legal_moves.append(action)
    return legal_moves