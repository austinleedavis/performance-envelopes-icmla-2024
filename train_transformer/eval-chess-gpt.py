# %%
from functools import partial
from typing import List

import chess
import datasets
import torch
from transformers import GPT2LMHeadModel

from modeling.chess_utils import uci_to_board
from modeling.linear_probe import *
from modeling.uci_tokenizers import UciTileTokenizer

NUM_LOGITS = 72


torch.set_grad_enabled(False)

DS_PATH: str = 'austindavis/lichess_uci'
DS_CONFIG_NAME: str = "201301-moves"  # dataset config
MODEL_PATH: str = "austindavis/chess-gpt2-uci-12x12x768"

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).requires_grad_(False)
model = model.to(torch.device("cuda"))
tokenizer = UciTileTokenizer()

ds = datasets.load_dataset(path=DS_PATH, name=DS_CONFIG_NAME, split="train")

MOVE_PHASES = [
        WHITE_TO,  # 2 phase = 0
        BLACK_FROM,  # 3 phase = 1
        BLACK_TO,  # 4 phase = 2
        WHITE_FROM,  # 1 phase = 3
        PROMOTION,  # 5 phase = 4
        SPECIAL,  # 6: note addressed here
    ] = range(6)

# %%

def diffy(x):
    return x[1] - x[0]

def get_board_fens_by_pos(transcript, offset_mapping):

    board_stack: List[chess.Board] = uci_to_board(
        transcript,
        as_board_stack=True,
        force=False,
        verbose=False,
    )

    fens_by_pos: List[str] = [board_stack[0].fen()]  # always include 1st board
    phases_by_pos: List[int] = [SPECIAL]
    # we must duplicate boards for each token. Sometimes, that's every 2 tokens,
    # sometimes that's every 3 (e.g., for promotions and at start tokens).
    board_pos = 0
    current_move_phase = 0
    for pos in range(1, len(offset_mapping)):
        if diffy(offset_mapping[pos]) < 2:
            phases_by_pos.append(PROMOTION)
        else:
            phases_by_pos.append(current_move_phase)
            current_move_phase = (current_move_phase + 1) % 4

        prior_offset = offset_mapping[pos - 1]
        current_offset = offset_mapping[pos]

        # check for space between previous token and current token
        if prior_offset[1] != current_offset[0]:
            board_pos += 1

        fens_by_pos.append(board_stack[board_pos].fen())

    return dict(fens_by_pos=fens_by_pos, phases_by_pos=phases_by_pos)


# Run only if the FEN dataset is not available
if BUILD_FEN_DATASET:= False:

    ds = (ds
        .map(
            lambda transcript: {"transcript": transcript.lower()},
            input_columns="transcript",
            )
        .map(
            lambda transcripts: tokenizer.batch_encode_plus(
        transcripts,
                return_token_type_ids=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_length=True,
            ),
            input_columns="transcript",
            batched=True,
            )
        .sort('length')
        .filter(lambda length: length < 300, input_columns='length')
        .select(range(100_000))
        .map(
            get_board_fens_by_pos,
            input_columns=['transcript', 'offset_mapping']
        )
    )

    ds.save_to_disk("data/201301-moves-fens_by_pos")

ds = datasets.load_from_disk("data/201301-moves-fens_by_pos")


# %%

forward = partial(
    model.__call__,
    use_cache=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
)


def do_forward(id_batch):
    padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in id_batch],
        batch_first=True,
    ).to(torch.device("cuda"))
    return dict(generated_ids=forward(padded)["logits"].argmax(-1).tolist())

# Run only if forward pass dataset is not available
if REBUILD_FORWARD_PASSES:=False:
    ds = ds.sort('length', reverse=True).map(
        do_forward,
        input_columns="input_ids",
        batched=True,
        batch_size=120,
        load_from_cache_file=False
    ).sort('length', reverse=False)

    ds.save_to_disk("data/201301-forward-passes")

ds = datasets.load_from_disk('data/201301-forward-passes')

# %%

def is_legal(fen, phase, id):
    board = chess.Board(fen)
    legal_moves = [
        (
            m.to_square
            if phase in [SPECIAL, WHITE_FROM, BLACK_FROM]
            else (m.from_square if phase in [WHITE_TO, BLACK_TO] else id)
        )
        for m in board.legal_moves
    ]
    return (id in legal_moves)

def count_legal_moves(fen_by_pos, phase_by_pos, id_by_pos):
    # Offset inputs because fens are incorrectly aligned to inputs
    fen_by_pos = fen_by_pos[1:]
    phase_by_pos = phase_by_pos[1:]
    id_by_pos = id_by_pos[:-1]
    legals = [
        is_legal(fen, phase, id-4)
        for (fen, phase, id) in zip(fen_by_pos, phase_by_pos, id_by_pos)
    ]
    return dict(count=sum(legals), total=len(legals), legals=legals)

if COMPUTE_LEGAL_MOVES:= True:
    ds = ds.map(
        count_legal_moves,
        input_columns=['fens_by_pos', 'phases_by_pos', 'generated_ids']
    )
    ds.save_to_disk("data/201301-forward-passes-with-legals")

ds = datasets.load_from_disk("data/201301-forward-passes-with-legals")
