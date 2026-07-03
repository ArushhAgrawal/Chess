import chess
import torch
from torch import nn
import numpy as np
import chess.pgn
#function to convert chess board to tensor
def board_to_tensor(board):
    #mapping
    piece_map = {
        chess.PAWN:0,
        chess.KNIGHT:1,
        chess.BISHOP:2,
        chess.ROOK:3,
        chess.QUEEN:4,
        chess.KING:5}
    chess_board= np.zeros((12,8,8),dtype=np.float32)
    #loop through all the squares on the chess board
    for square in chess.SQUARES:
        piece= board.piece_at(square)
        if piece is not None:
            r= chess.square_rank(square)
            c=chess.square_file(square)

            #finding out which peice belong where
            piece_layer= piece_map[piece.piece_type]
            if piece.color==chess.BLACK:
                piece_layer+=6
            chess_board[piece_layer, r,c]=1.0
    return torch.Tensor(chess_board)

#encoding moves, giving them a unique number for model to understand later
def encoding(move):
    #formula we use is (start *64)+end
    return (move.from_square * 64)+ move.to_square

# chess_board= chess.Board()
# tensor_chess_board= board_to_tensor(chess_board)
# print(tensor_chess_board)

#writing boiler plate stuff
class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.convstack= nn.Sequential(
            nn.Conv2d(12,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),#comprises the input to a smaller size 
            nn.ReLU(),
            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8*8, 4096),
            nn.ReLU()
        )
    def forward(self, x):
        return self.classifier(self.convstack(x))
    