from typing import List

import chess
import tokenizers
from tokenizers import models, pre_tokenizers, processors
from torch import Tensor as TT
from transformers import PreTrainedTokenizerFast


class UciTokenizerBase(PreTrainedTokenizerFast):
    """Base class for UCI tokenizers. Should not be instantiated."""

    _PAD_TOKEN: str
    _UNK_TOKEN: str
    _EOS_TOKEN: str
    _BOS_TOKEN: str

    stoi: dict[str, int]
    """Integer to String mapping"""

    itos: dict[int, str]
    """String to Integer Mapping. This is the vocab"""

    def __init__(
        self,
        stoi,
        itos,
        pad_token,
        unk_token,
        bos_token,
        eos_token,
        name_or_path,
        **kwargs,
    ):
        self.stoi = stoi
        self.itos = itos

        self._PAD_TOKEN = pad_token
        self._UNK_TOKEN = unk_token
        self._EOS_TOKEN = eos_token
        self._BOS_TOKEN = bos_token

        # Define the model
        tok_model = models.WordLevel(vocab=self.stoi, unk_token=self._UNK_TOKEN)

        slow_tokenizer = tokenizers.Tokenizer(tok_model)
        slow_tokenizer.pre_tokenizer = self._init_pretokenizer()

        # post processing adds special tokens unless explicitly ignored
        post_proc = processors.TemplateProcessing(
            single=f"{bos_token} $0",
            pair=None,
            special_tokens=[(bos_token, 1)],
        )
        slow_tokenizer.post_processor = post_proc

        super().__init__(
            tokenizer_object=slow_tokenizer,
            unk_token=self._UNK_TOKEN,
            bos_token=self._BOS_TOKEN,
            eos_token=self._EOS_TOKEN,
            pad_token=self._PAD_TOKEN,
            name_or_path=name_or_path,
            **kwargs,
        )

        # Override the decode behavior to ensure spaces are correctly handled
        def _decode(
            token_ids: int | List[int],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) -> int | List[int]:

            if isinstance(token_ids, int):
                return self.itos.get(token_ids, self._UNK_TOKEN)

            if isinstance(token_ids, dict):
                token_ids = token_ids["input_ids"]

            if isinstance(token_ids, TT):
                token_ids = token_ids.tolist()

            if isinstance(token_ids, list):
                tokens_str = [self.itos.get(xi, self._UNK_TOKEN) for xi in token_ids]
                moves = self._process_str_tokens(tokens_str)

                return " ".join(moves)

        self._decode = _decode

    def _init_pretokenizer(self) -> pre_tokenizers.PreTokenizer:
        raise NotImplementedError

    def _process_str_tokens(self, tokens_str: list[str]) -> list[str]:
        raise NotImplementedError

    def get_id2square_list() -> list[int]:
        raise NotImplementedError


class LcbTokenizer(UciTokenizerBase):
    """
    Reimplementation of Toshniwal's tokenizer from 'Chess as a testbed for Language Model State Tracking'.
    This UCI tokenizer converting start/end tiles and promotion types each into individual tokens.
    This is nearly identical to the `UciTileTokenizer`, except it adds 6 tokens (one fore each piece type)
    at IDs 3 throguh 8 (PNRBQK)."""

    stoi = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "P": 3,
        "N": 4,
        "R": 5,
        "B": 6,
        "Q": 7,
        "K": 8,
        "c2": 9,
        "c4": 10,
        "e7": 11,
        "e5": 12,
        "g2": 13,
        "g3": 14,
        "b8": 15,
        "c6": 16,
        "f1": 17,
        "g8": 18,
        "f6": 19,
        "b1": 20,
        "c3": 21,
        "f8": 22,
        "b4": 23,
        "d5": 24,
        "e8": 25,
        "e2": 26,
        "e3": 27,
        "e4": 28,
        "a2": 29,
        "a3": 30,
        "d6": 31,
        "d8": 32,
        "a1": 33,
        "e6": 34,
        "b2": 35,
        "b3": 36,
        "c7": 37,
        "d2": 38,
        "d3": 39,
        "f7": 40,
        "f5": 41,
        "a7": 42,
        "a5": 43,
        "g1": 44,
        "d4": 45,
        "g4": 46,
        "c5": 47,
        "e1": 48,
        "d7": 49,
        "d1": 50,
        "h8": 51,
        "c1": 52,
        "a4": 53,
        "h2": 54,
        "h3": 55,
        "a8": 56,
        "f4": 57,
        "g7": 58,
        "g5": 59,
        "b6": 60,
        "b7": 61,
        "c8": 62,
        "h7": 63,
        "h4": 64,
        "f3": 65,
        "h5": 66,
        "h1": 67,
        "f2": 68,
        "h6": 69,
        "g6": 70,
        "b5": 71,
        "a6": 72,
        "q": 73,
        "r": 74,
        "b": 75,
        "n": 76,
    }
    itos = {
        0: "<pad>",
        1: "<s>",
        2: "</s>",
        3: "P",
        4: "N",
        5: "R",
        6: "B",
        7: "Q",
        8: "K",
        9: "c2",
        10: "c4",
        11: "e7",
        12: "e5",
        13: "g2",
        14: "g3",
        15: "b8",
        16: "c6",
        17: "f1",
        18: "g8",
        19: "f6",
        20: "b1",
        21: "c3",
        22: "f8",
        23: "b4",
        24: "d5",
        25: "e8",
        26: "e2",
        27: "e3",
        28: "e4",
        29: "a2",
        30: "a3",
        31: "d6",
        32: "d8",
        33: "a1",
        34: "e6",
        35: "b2",
        36: "b3",
        37: "c7",
        38: "d2",
        39: "d3",
        40: "f7",
        41: "f5",
        42: "a7",
        43: "a5",
        44: "g1",
        45: "d4",
        46: "g4",
        47: "c5",
        48: "e1",
        49: "d7",
        50: "d1",
        51: "h8",
        52: "c1",
        53: "a4",
        54: "h2",
        55: "h3",
        56: "a8",
        57: "f4",
        58: "g7",
        59: "g5",
        60: "b6",
        61: "b7",
        62: "c8",
        63: "h7",
        64: "h4",
        65: "f3",
        66: "h5",
        67: "h1",
        68: "f2",
        69: "h6",
        70: "g6",
        71: "b5",
        72: "a6",
        73: "q",
        74: "r",
        75: "b",
        76: "n",
    }

    id2square = [
        33,
        20,
        52,
        50,
        48,
        17,
        44,
        67,
        29,
        35,
        9,
        38,
        26,
        68,
        13,
        54,
        30,
        36,
        21,
        39,
        27,
        65,
        14,
        55,
        53,
        23,
        10,
        45,
        28,
        57,
        46,
        64,
        43,
        71,
        47,
        24,
        12,
        41,
        59,
        66,
        72,
        60,
        16,
        31,
        34,
        19,
        70,
        69,
        42,
        61,
        37,
        49,
        11,
        40,
        58,
        63,
        56,
        15,
        62,
        32,
        25,
        22,
        18,
        51,
    ]

    def get_id2square_list(self) -> List[int]:
        return self.id2square

    def __init__(self):

        super().__init__(
            self.stoi,
            self.itos,
            pad_token="<pad>",
            unk_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            name_or_path="austindavis/uci_tile_tokenizer",
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(r"\d")
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str):
        moves = []
        next_move = ""
        for token in token_str:

            # skip special tokens
            if token in self.all_special_tokens:
                continue

            # handle promotions
            if len(token) == 1:
                moves.append(next_move + token)
                continue

            # handle regular tokens
            if len(next_move) == 4:
                moves.append(next_move)
                next_move = token
            else:
                next_move += token

        moves.append(next_move)
        return moves


class UciTileTokenizer(UciTokenizerBase):
    """Uci tokenizer converting start/end tiles and promotion types each into individual tokens"""

    stoi = {
        tok: idx
        for tok, idx in list(
            zip(
                ["<pad>", "<s>", "</s>", "<unk>"] + chess.SQUARE_NAMES + list("qrbn"),
                range(72),
            )
        )
    }

    itos = {
        idx: tok
        for tok, idx in list(
            zip(
                ["<pad>", "<s>", "</s>", "<unk>"] + chess.SQUARE_NAMES + list("qrbn"),
                range(72),
            )
        )
    }

    id2square: List[int] = list(range(4, 68))
    """
    List mapping token IDs to squares on the chess board. Order is file then row, i.e.: 
    `A1, B1, C1, ..., F8, G8, H8`    
    """

    def get_id2square_list(self) -> List[int]:
        return self.id2square

    def __init__(self, **kwargs):

        super().__init__(
            self.stoi,
            self.itos,
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            name_or_path="austindavis/uci_tile_tokenizer",
            **kwargs,
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(r"\d")
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str):
        moves = []
        next_move = ""
        for token in token_str:

            # skip special tokens
            if token in self.all_special_tokens:
                continue

            # handle promotions
            if len(token) == 1:
                moves.append(next_move + token)
                continue

            # handle regular tokens
            if len(next_move) == 4:
                moves.append(next_move)
                next_move = token
            else:
                next_move += token

        moves.append(next_move)
        return moves


class UciCharTokenizer(UciTokenizerBase):
    """Uci tokenizer converting every character into a token"""

    itos = {
        0: " ",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: ";",
        10: "#",
        11: "a",
        12: "b",
        13: "c",
        14: "d",
        15: "e",
        16: "f",
        17: "g",
        18: "h",
        19: "n",
        20: "r",
        21: "q",
        22: "k",
    }
    """Integer to String mapping"""

    stoi = {
        " ": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        ";": 9,
        "#": 10,
        "a": 11,
        "b": 12,
        "c": 13,
        "d": 14,
        "e": 15,
        "f": 16,
        "g": 17,
        "h": 18,
        "n": 19,
        "r": 20,
        "q": 21,
        "k": 22,
    }
    """String to Integer Mapping. This is the vocab"""

    def __init__(self):
        super().__init__(
            self.stoi,
            self.itos,
            pad_token="<pad>",
            unk_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
            name_or_path="austindavis/uci_char_tokenizer",
        )

    def _init_pretokenizer(self):
        # Pre-tokenizer to split input into UCI moves
        pattern = tokenizers.Regex(r".")
        pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Whitespace(),
                pre_tokenizers.Split(pattern=pattern, behavior="merged_with_previous"),
            ]
        )
        return pre_tokenizer

    def _process_str_tokens(self, token_str):
        moves = []
        next_move = ""
        for token in token_str:

            # skip special tokens
            if token in self.all_special_tokens:
                continue

            if len(next_move) <= 4:
                next_move += token
                continue

            # move length is now 5

            # handle easy promotes
            if next_move[-1] in "nrqk":
                moves.append(next_move)
                next_move = token
                continue

            # handle bishop promotes
            if next_move[-1] == "b" and token in "abcdefgh":
                moves.append(next_move)
                next_move = token
                continue

            # non-promotion, clear next move
            moves.append(next_move[:-1])
            next_move = next_move[-1] + token

        moves.append(next_move)
        return moves


class UciMoveTokenizer(UciTokenizerBase):
    """
    A UCI move tokenizer. Computed vocabulary includes all possible UCI moves. 
    The vocabulary includes all 1857 possible legal UCI moves in chess.

    The moves are computed by placing a piece on a tile in an otherwise empty board then computing 
    the legal moves from that position. These legal moves are added to a set to filter out 
    duplicates. Since queens can move diagonally, horizontally, and vertically, their legal moves 
    already capture all possible moves of the bishops, rooks, kings, and most possible pawn moves. 
    Pawn promotions (which include a 5th letter to indicate the promotion type) are are computed 
    separately, as are all moves by the knights (since their movement pattern is unlike a queen).
    """

    __UNK_TOKEN__: str = "<UNK>"
    __BOS_TOKEN__: str = ";"
    __EOS_TOKEN__: str = "<EOS>"

    stoi: dict = {
        "#": 0,
        __UNK_TOKEN__: 1,
        __BOS_TOKEN__: 2,
        __EOS_TOKEN__: 3,
        # PAD TOKEN SET TO EOS TOKEN
    }

    itos: dict

    def __init__(self):
        self.make_vocab()
        super().__init__(
            self.stoi,
            self.itos,
            pad_token=self.__EOS_TOKEN__,
            unk_token=self.__UNK_TOKEN__,
            bos_token=self.__BOS_TOKEN__,
            eos_token=self.__EOS_TOKEN__,
            name_or_path="austindavis/uci_move_tokenizer",
        )

    def _init_pretokenizer(self):
        return pre_tokenizers.Whitespace()

    def _process_str_tokens(self, token_str):
        moves = [token for token in token_str if token not in self.all_special_tokens]
        return moves

    def make_vocab(self):
        move_set = set()

        # compute all possible moves for a queen (we get bishop, king, rook moves for free)
        for i in range(64):
            board = chess.Board()
            board = board.empty()
            board.set_piece_at(
                chess.square(i % 8, i // 8), chess.Piece(chess.QUEEN, chess.WHITE)
            )
            legal_moves = board.generate_legal_moves()
            for move in legal_moves:
                move_set.add(move.uci())

        # compute all possible moves for a knight
        for i in range(64):
            board = chess.Board()
            board = board.empty()
            board.set_piece_at(
                chess.square(i % 8, i // 8), chess.Piece(chess.KNIGHT, chess.WHITE)
            )
            legal_moves = board.generate_legal_moves()
            for move in legal_moves:
                move_set.add(move.uci())

        # compute all possible promotion moves for white pawns
        white_pawn_squares = [chess.SQUARES[i] for i in range(48, 56)]
        for sq in white_pawn_squares:
            board = chess.Board()
            board = board.empty()
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.WHITE))
            legal_moves = board.generate_legal_moves()
            for move in legal_moves:
                move_set.add(move.uci())

        # compute all possible promotion moves for black pawns
        black_pawn_squares = [chess.SQUARES_180[i] for i in range(48, 56)]
        for sq in black_pawn_squares:
            board = chess.Board()
            board = board.empty()
            board.turn = chess.BLACK
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
            legal_moves = board.generate_legal_moves()
            for move in legal_moves:
                move_set.add(move.uci())

        # define the string2index and index2string dicts
        # vocab keys are sorted alphabetically to make it easier to reproduce

        self.stoi.update(
            {
                uci: idx + len(self.stoi.keys())
                for idx, uci in enumerate(sorted(list(move_set)))
            }
        )
        self.itos = {self.stoi[uci]: uci for uci in self.stoi.keys()}


if __name__ == "__main__":
    for tok in [
        UciCharTokenizer(),
        UciMoveTokenizer(),
        LcbTokenizer(),
        UciTileTokenizer(),
    ]:
        test_str = "b7b8b f7f5 a2a1r d7d6 e2e1n b7b5 d7e8q e7e6 e2d1 f5f4 g1f3 f8e7 h2h3 b5b4 g2g4 f4g3 d2d4 c8a6 c3a4 d8c8 f3h2 e6e5 f1b5 e8f8 h2f3 a6b7 d1d2 e5d4 b5e8 c8e6 e8h5 e7f6 d2e2 e6c4 e1d1 c4e6 h5g4 e6d7 e2d2 g3f2 d2e2 a7a5 d1d2 f2f1q e2h2 b7d5 a2a3 b4a3 c2c4 f6e5 c4c5 b8a6 h1f1 d5f7 f1d1 d7e7 d2e2 e7f6 h2e5 a6b4 e3d4 f7d5 g4c8 g8e7 c8a6 f6h6 a6b5 d6e5 a4c3 f8"
        print(tok, "\n")
        print(tok(test_str), "\n")
        print(tok.decode(tok(test_str)), "\n")
