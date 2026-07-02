import chess
import torch
from torch import nn
import numpy as np

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
    #formula we use is (start *63)+end
    return (move.from_square * 64)+ move.to_square

chess_board= chess.Board()
tensor_chess_board= board_to_tensor(chess_board)
print(tensor_chess_board)