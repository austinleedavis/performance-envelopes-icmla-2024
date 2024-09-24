"""
board state functions map a dataframe containing uci strings to one hot encodings for training probes.

The primary function is `df_to_one_hot_tensor`. It does most of the heavy lifting with the tensors and 
requires a callable `state_mapper` which converts a game (in fen_stack format) into a state vector.
state_mappers must return a 2D list with shape [n_classes, 64].

One thing to keep in mind is we are mapping moves to state. We are not mapping tokens to state.
"""


# %%
from typing import Any, Callable

import pandas as pd
import torch

import chess


class StateFunctionBase:
    def map():
        ...

    @classmethod
    def get_class_labels(cls):
        return cls.class_labels


class BoardStateFunctionBase(StateFunctionBase):

    @classmethod
    def diff(cls, board1: chess.Board, board2: chess.Board):
        """
        Differences two chess boards. Outputs a dictionary with key-value pairs:
            - names list[str]: Tile names that are different
            - indices list[int]: Tile indices that are different
            - add list[int]: In the order matching `indices`, the class id which was added to each tile.
            - remove list[int]: Sorted according to the
        
        """
        # delta.shape is [64, num_classes]
        delta = (cls.apply(board1)-cls.apply(board2)).permute(1,0)
        mask = ~(delta == torch.zeros([64, cls.num_classes])).all(dim=1)
        tile_changes = delta[mask]
        indices = torch.arange(64)[mask].tolist()
        squares = [chess.SQUARE_NAMES[i] for i in indices]
        adds = torch.nonzero(tile_changes == -1, as_tuple=False).T[1].tolist()
        subtracts = torch.nonzero(tile_changes == 1, as_tuple=False).T[1].tolist()
        labels = cls.get_class_labels()
        result = [
            {
                "ids": {
                    "square": indices[i],
                    "add": adds[i],
                    "subtract": subtracts[i],
                },
                "text": {
                    "square": squares[i],
                    "add": labels[adds[i]],
                    "subtract": labels[subtracts[i]],
                },
            }
            for i in range(len(squares))
        ]
        return result

    @classmethod
    def map(cls, dataset_sample: Any, column_name: str):
        return cls.apply(chess.Board(dataset_sample[column_name]))

class ToColor(BoardStateFunctionBase):
    num_classes = 3
    class_labels = ["Black", "White", "⦰"]

    @staticmethod
    def apply(board: chess.Board):
        return torch.tensor(to_color(board))


class ToPiece(BoardStateFunctionBase):
    num_classes = 7
    class_labels = ['P', 'N', 'B', 'R', 'Q', 'K', '⦰']

    @staticmethod
    def apply(board: chess.Board):
        return torch.tensor(to_piece(board))


class ToColorFlipping(BoardStateFunctionBase):
    num_classes = 3
    class_labels = ['mine', 'yours', 'empty']
    @staticmethod
    def apply(board: chess.Board):
        return torch.tensor(to_color_flipping(board))

class ToPieceByColor(BoardStateFunctionBase):
    num_classes = 13
    class_labels = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K', '⦰']
    
    @staticmethod
    def apply(board: chess.Board):
        return torch.tensor(to_piece_by_color(board))


def to_color(board: chess.Board):
    """
    Converts the chess board state into a single token color state representation.
    Suitable when slicing every 4th token.

    Args:
        board (chess.Board): The chess board state.

    Returns:
        list: A list containing the binary representation of the occupied squares for white pieces,
              black pieces, and blank squares, respectively.
    """

    mask_b = board.occupied_co[chess.BLACK]
    mask_w = board.occupied_co[chess.WHITE]

    blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF

    single_token_state = [
        _bit_mask_to_bit_list(mask_b),
        _bit_mask_to_bit_list(mask_w),
        _bit_mask_to_bit_list(blanks),
    ]
    return single_token_state


def to_color_flipping(board: chess.Board):
    """
    Converts the chess board state into a single token color state,
    flipping the tile mask based on the current player.
    Suitable when slicing every 2nd token

    n_classes = 3

    Args:
        board (chess.Board): The chess board object representing the current state.

    Returns:
        list: A list containing the binary representation of the white pieces, black pieces, and blank tiles.
    """

    mask_b = board.occupied_co[chess.BLACK]
    mask_w = board.occupied_co[chess.WHITE]

    # This flips the tile mask based on the current player
    if not board.turn:
        temp = mask_w
        mask_w = mask_b
        mask_b = temp

    blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF

    single_token_state = [
        _bit_mask_to_bit_list(mask_b),
        _bit_mask_to_bit_list(mask_w),
        _bit_mask_to_bit_list(blanks),
    ]
    return single_token_state


def to_piece(board: chess.Board):
    """
    Converts the given chess board state into a one-hot vector representation of piece state.
    n_clases = 7 (one for each piece type)

    Args:
        board (chess.Board): The chess board state.

    Returns:
        list: A list representing the one-hot vector representation of piece state, where each element
              corresponds to a different piece type in the following order:
              - pawns
              - knights
              - bishops
              - rooks
              - queens
              - kings

    """
    mask_b = board.occupied_co[chess.BLACK]
    mask_w = board.occupied_co[chess.WHITE]

    blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF

    single_token_state = [
        _bit_mask_to_bit_list(board.pawns),
        _bit_mask_to_bit_list(board.knights),
        _bit_mask_to_bit_list(board.bishops),
        _bit_mask_to_bit_list(board.rooks),
        _bit_mask_to_bit_list(board.queens),
        _bit_mask_to_bit_list(board.kings),
        _bit_mask_to_bit_list(blanks),
    ]
    return single_token_state


def to_piece_by_color(board: chess.Board):
    """
    Converts the chess board state into a one-hot vector representation of piece type and color for each tile on the grid.
    n_classes = 13

    Args:
        board (chess.Board): The chess board state.

    Returns:
        list: A list representing the one-hot state, where each element corresponds to a piece type and color.
              The list contains the following bitlists in order:
              - white pawns
              - white knights
              - white bishops
              - white rooks
              - white queens
              - white kings
              - black pawns
              - black knights
              - black bishops
              - black rooks
              - black queens
              - black kings
              - blank squares
    """
    mask_b = board.occupied_co[chess.BLACK]
    mask_w = board.occupied_co[chess.WHITE]

    blanks = ~(mask_w | mask_b) & 0xFFFFFFFFFFFFFFFF

    single_token_state = [
        # black pieces
        _bit_mask_to_bit_list(board.pawns & mask_b),
        _bit_mask_to_bit_list(board.knights & mask_b),
        _bit_mask_to_bit_list(board.bishops & mask_b),
        _bit_mask_to_bit_list(board.rooks & mask_b),
        _bit_mask_to_bit_list(board.queens & mask_b),
        _bit_mask_to_bit_list(board.kings & mask_b),
        # white pieces
        _bit_mask_to_bit_list(board.pawns & mask_w),
        _bit_mask_to_bit_list(board.knights & mask_w),
        _bit_mask_to_bit_list(board.bishops & mask_w),
        _bit_mask_to_bit_list(board.rooks & mask_w),
        _bit_mask_to_bit_list(board.queens & mask_w),
        _bit_mask_to_bit_list(board.kings & mask_w),
        # blanks
        _bit_mask_to_bit_list(blanks),
    ]
    return single_token_state


def to_my_controlled_tiles(board: chess.Board):
    """highlights tiles that are reachable (i.e., either being defended or attacked) by white"""
    attackers_list = []

    flip = board.turn
    
    for tile in range(64):
        mask = board.attackers_mask(flip, tile)

        attackers_list.append(_bit_mask_to_bit_list(mask))

    return attackers_list


def to_their_controlled_tiles(board: chess.Board):
    """highlights tiles that are reachable (i.e., either being defended or attacked) by white"""
    attackers_list = []
    
    flip = ~board.turn

    for tile in range(64):
        mask = board.attackers_mask(flip, tile)

        attackers_list.append(_bit_mask_to_bit_list(mask))

    return attackers_list


def to_black_controlled_tiles(board: chess.Board):
    """Reachable by"""
    attackers_list = []
    for tile in range(64):
        mask = board.attackers_mask(0, tile)

        attackers_list.append(_bit_mask_to_bit_list(mask))

    return attackers_list


def to_white_controlled_tiles(board: chess.Board):
    """highlights tiles that are reachable (i.e., either being defended or attacked) by white"""
    attackers_list = []
    for tile in range(64):
        mask = board.attackers_mask(1, tile)
        # mask |= board.attackers_mask(1,tile)

        attackers_list.append(_bit_mask_to_bit_list(mask))

    return attackers_list


def to_black_controlled_tiles(board: chess.Board):
    """Reachable by"""
    attackers_list = []
    for tile in range(64):
        mask = board.attackers_mask(0, tile)

        attackers_list.append(_bit_mask_to_bit_list(mask))

    return attackers_list


def df_to_one_hot_tensor(
    df: pd.DataFrame, state_mapper: Callable[..., list], move_based=True
):
    """
    Convert a DataFrame of chess board uci strings to a one-hot tensor representation.

    Args:
        df (pd.DataFrame): The DataFrame containing the chess board states.
        state_function (Callable): A function that takes a chess.Board object as input and returns a list of state vectors. The shape should be [n_classes, 64].

    Returns:
        torch.Tensor: The one-hot tensor representation of the chess board states.

    """
    state_lists = []
    for _, row in df[["fen_stack", "input_ids"]].iterrows():
        board_state_list = []
        for fen_str in row["fen_stack"]:
            board = chess.Board(fen_str)
            board_state_vector_list = state_mapper(board)  # len=n_classes
            
            # ---------
            # state_mapper returns list or a tuple of lists.
            # append twice (extend) if state_mapper is move-based (tuple retun)
            # append once if state_mapper is token-based
            # ---------
            
            # _append_item_or_items(board_state_list, board_state_vector_list)
            board_state_list.append(board_state_vector_list)
            board_state_list.append(board_state_vector_list)
            

        state_lists.append(board_state_list)

    state_tensor = torch.tensor(state_lists, dtype=torch.float32, device="cuda:0")

    n_classes = len(state_lists[-1][-1])
    n_pos = len(row["input_ids"])
    batch_size = len(df)

    state_tensor = state_tensor.reshape(batch_size, n_pos, n_classes, 8, 8)

    return state_tensor.permute(0, 1, 3, 4, 2).flip(-2)


def _bit_mask_to_bit_list(value: int):
    """
    Converts an integer value into a list of its binary representation.

    Args:
        value (int): The integer value to convert.

    Returns:
        list: A list of 64 elements representing the binary representation of the value.
    """
    bits = [(value >> i) & 1 for i in range(0, 64, 1)]
    return bits


def _append_item_or_items(lst, item):
    if isinstance(item, tuple):
        lst.append(item[0])
        lst.append(item[1])
    else:
        lst.append(item)


# %%
if __name__ == "__main__":
    # do Run Above!

    import plotly.express as px
    from IPython.display import display

    print(torch.tensor(to_piece(chess.Board())).shape)

    move_id = 0
    df = pd.read_pickle("chess_data/lichess_test.pkl")
    print(len(df), df.keys(), sep="\n")
    small_df = df.iloc[0:1]

    display(chess.Board(small_df["fen_stack"].iloc[game_id:=0][move_id]))
    # ----------
    # Working Examples below Uncomment to run
    # -----------
    # one_hots = df_to_one_hot_tensor(small_df,to_piece)
    # display(px.imshow(one_hots[game_id,move_id*2].cpu(), animation_frame = -1, title="to_piece (one pos, varying class)"))

    # one_hots = df_to_one_hot_tensor(small_df,to_color)
    # display(px.imshow(one_hots[game_id,move_id*2].cpu(), animation_frame = -1, title = "to_color (one pos, varying class)"))

    # one_hots = df_to_one_hot_tensor(small_df,to_piece_by_color)
    # display(px.imshow(one_hots[game_id,move_id*2].cpu(), animation_frame = -1, title = "to_piece_by_color (one pos, varying class)"))

    # one_hots = df_to_one_hot_tensor(small_df,to_color_flipping)
    # print(one_hots.shape)
    # display(px.imshow(one_hots[game_id,...,0].cpu(), animation_frame = 0, title = "to_color_flipping (one class, varying pos)"))

    # one_hots = df_to_one_hot_tensor(small_df,to_white_controlled_tiles)
    # display(px.imshow(one_hots[game_id,move_id*2].cpu(), animation_frame = -1, title="white controlled (one move, varying tile from BL to TR)"))

    one_hots = df_to_one_hot_tensor(small_df,to_my_controlled_tiles)
    display(px.imshow(one_hots[game_id,move_id*2].cpu(), animation_frame = -1, title="my controlled (one move, varying tile from BL to TR)"))
    display(px.imshow(one_hots[game_id,(move_id+1)*2].cpu(), animation_frame = -1, title="their controlled (one move, varying tile from BL to TR)"))


# %%
