# CheckersAgent
Checkers Deep Neural Network agent 


# Implementation Notes
- Board (observation space) implemented as 8x8 matrix, -2 is opp king, -1 is opp piece, 0 is empty square, 1 is agent piece, 2 is agent king.
- action space is definced by 32x8 space (choice between 256 possible moves, not all of which are legal) from square, and direction
- mask should check each move and choose whether it is legal or not, 1 if legal and 0 if illegal
- 
Questions:
- Should forced moves be made automatically or should they just be masked?
- 