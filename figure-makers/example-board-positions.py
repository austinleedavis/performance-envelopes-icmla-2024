""" Creates the 'Example Board Positions' figure."""
# %%
import chess
import chess.svg
import torch
from IPython.display import display
from ipywidgets import GridspecLayout, Output
from torch import Tensor
from transformers import GPT2LMHeadModel

from modeling.chess_utils import uci_to_board
from modeling.uci_tokenizers import UciTileTokenizer

tokenizer = UciTileTokenizer()

ds = [
    {
        "site": "j1dkb5dw",
        "transcript": "e2e4 e7e6 d2d4 b7b6 a2a3 c8b7 b1c3 g8h6 c1h6 g7h6 f1e2 d8g5 e2g4 h6h5 g1f3 g5g6 f3h4 g6g5 g4h5 g5h4 d1f3 e8d8 f3f7 b8c6 f7e8",
    }
]


model = GPT2LMHeadModel.from_pretrained(
    "austindavis/chess-gpt2-uci-12x12x768",
)

with torch.no_grad():
    logits: Tensor = model(
        tokenizer.batch_encode_plus(
            [ds[0]["transcript"]],
            return_tensors="pt",
        )["input_ids"]
    )[0].squeeze()

probs = logits.softmax(-1)

output_ids = logits.argmax(-1)

output_str = tokenizer.batch_decode(output_ids)

board_stack = uci_to_board(ds[0]["transcript"], as_board_stack=True)

idx = 8
original = board_stack[idx]
intervention = original.copy()
intervention.remove_piece_at(chess.SQUARES[2])
future = board_stack[idx+6]

SIZE = 400
COLORS = {'square dark':"#EEEEEB",
          'square light':"#A8A8A8"}

boards = []

boards.append(
    chess.svg.board(
        original,
        arrows=[chess.svg.Arrow(chess.C1, head=chess.H6)],
        size=SIZE,
        colors=COLORS,
        borders=True,
        coordinates=False,
    )
)

boards.append(
    chess.svg.board(
        intervention,
        squares=[2],
        check=chess.SQUARES[2],
        size=SIZE,
        colors=COLORS,
        borders=True,
        coordinates=False,
    )
)
boards.append(
    chess.svg.board(
        intervention,
        arrows=[chess.svg.Arrow(tail=chess.G1, head=chess.F3)],
        size=SIZE,
        colors=COLORS,
        borders=True,
        coordinates=False,
    )
)
boards.append(
    chess.svg.board(
        future,
        # arrows=[chess.svg.Arrow(tail=chess.G1, head=chess.F3)],
        size=SIZE,
        colors=COLORS,
        borders=True,
        coordinates=False,
    )
)



grid = GridspecLayout(2, 2)
for i, board in enumerate(boards):
    out=Output()
    with out:
        display(board)
    grid[i//2,i%2]=out

grid