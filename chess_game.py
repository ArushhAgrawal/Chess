import chess
import torch
from torch import nn
import numpy as np
import chess.pgn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset as td
import os
import time
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
    chess_board= np.zeros((12,8,8),dtype=np.float32)#changed to 32 bit directly
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
    return chess_board

#encoding moves, giving them a unique number for model to understand later
def encoding(move):
    #formula we use is (start *64)+end
    return (move.from_square * 64)+ move.to_square

# chess_board= chess.Board()
# tensor_chess_board= board_to_tensor(chess_board)
# print(tensor_chess_board)

#preparing the dataset
def prepare_dataset(pgn_file,start_game, end_game):
    x=[]
    y=[]
    with open(pgn_file, "r", encoding="utf-8", errors="replace") as pgn:
        game_count=0
        while game_count<end_game:
            game= chess.pgn.read_game(pgn)
            if game is None:
                break
            if game_count>=start_game and game_count<end_game:
                board= game.board()
                for move in game.mainline_moves():#game.mainline_moves() gives us the moves in the game
                    x.append(torch.from_numpy(board_to_tensor(board)))#changed it to torch.from_numpy better performance
                    y.append(encoding(move))
                    board.push(move)#board.push(move) updates the board with the move that was just made
            game_count+=1
    x_tensor= torch.stack(x)
    y_tensor= torch.tensor(y, dtype=torch.int64)
    return x_tensor, y_tensor

#giving dataset to the model
#raw datset
train_cache= "train_cache.pt"
test_cache="test_cache.pt"
if not os.path.exists(train_cache):
    print("Preparing training dataset...")
    x_train, y_train= prepare_dataset("Modern.pgn", 0, 14000)
    x_test, y_test= prepare_dataset("Modern.pgn", 14000, 17000)
    print("saving training dataset to cache...")
    torch.save((x_train.to(torch.uint8),y_train), train_cache)
    torch.save((x_test.to(torch.uint8),y_test), test_cache)
else:
    print("Loading training dataset from cache...")
    x_train, y_train= torch.load(train_cache)
    x_test, y_test= torch.load(test_cache)
chess_train_dataset= td(x_train, y_train)#dataset in tensor format
train_dataloader= DataLoader(chess_train_dataset, batch_size=512, shuffle=True)#loading the dataset
chess_test_dataset= td(x_test, y_test)
test_dataloader= DataLoader(chess_test_dataset, batch_size=512, shuffle=False)

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
            nn.Conv2d(128,64, kernel_size=3, padding=1),#using 3x3 lower decent 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,2, kernel_size=1),
            nn.BatchNorm2d(2),
        )
        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*8*8, 4096)
        )
    def forward(self, x):
        return self.classifier(self.convstack(x))
model= ChessModel().to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
# print(model.parameters())

#training the model
def train_model(model,train,test,loss_fn,optimizer,epochs):
    torch.mps.manual_seed(32)
    start= time.time()
    for epoch in range(epochs):
        model.train()
        for batch, (x,y) in enumerate(train):
            x=x.to(device, dtype=torch.float32)
            y=y.to(device, dtype=torch.int64)
            y_logit_train=model(x)
            loss_train=loss_fn(y_logit_train, y)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            for batch, (x,y) in enumerate(test):
                x=x.to(device, dtype=torch.float32)
                y=y.to(device, dtype= torch.int64)
                y_logit_test=model(x)
                loss_test=loss_fn(y_logit_test, y)
    end=time.time()
    return loss_train, loss_test, end-start
print(train_model(model,train_dataloader,test_dataloader,loss_fn,optimizer,epochs=4))