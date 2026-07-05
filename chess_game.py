import chess
import torch
from torch import nn
import numpy as np
import chess.pgn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset as td
#device agnostic code
device= "mps" if torch.mps.is_available() else "cpu"

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

#preparing the dataset
def prepare_dataset(pgn_file,max_game):
    x=[]
    y=[]
    with open(pgn_file, "r") as pgn:
        game_count=0
        while game_count<max_game:
            game= chess.pgn.read_game(pgn)
            if game is None:
                break
            board= game.board()
            for move in game.mainline_moves():#game.mainline_moves() gives us the moves in the game
                x.append(board_to_tensor(board))
                y.append(encoding(move))
                board.push(move)#board.push(move) updates the board with the move that was just made
            game_count+=1
    x_tensor= torch.stack(x)
    y_tensor= torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor

#giving dataset to the model
all_x, all_y= prepare_dataset("Modern.pgn", 17000)#raw datset
x_train, y_train= all_x[:14000], all_y[:14000]#raw datset
chess_train_dataset= td(x_train, y_train)#dataset in tensor format
train_dataloader= DataLoader(chess_train_dataset, batch_size=32, shuffle=True)#loading the dataset

x_test, y_test= all_x[14000:], all_y[14000:]#raw datset(slicing it so we can skip data repetition)
chess_test_dataset= td(x_test, y_test)
test_dataloader= DataLoader(chess_test_dataset, batch_size=32, shuffle=True)

# t,f= next(iter(train_dataloader
# print(t.shape)

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
model= ChessModel()
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
# print(model.parameters())

#training the model
def train_model(model,x,y,loss_fn,optimizer,epochs):
    torch.mps.manual_seed(32)
    for epoch in range(epochs):
        model.train()
        y_logit_train=model(x)
        loss=loss_fn(y_logit_train, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
