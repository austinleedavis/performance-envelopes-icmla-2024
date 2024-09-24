from typing import Any, Callable, Iterable, List, Union

import chess
import chess.svg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from IPython.display import display


def pretty_moves(uci: str, wrap_width: int = 4):
    """
    Displays a sequence of UCI moves highlighted using either black or white depending on 
    which players makes the move. Assumes, the first move is made by white.
    """
    uci_moves = uci.split(" ")
    colors = [
        "\u001b[30;47m" if i % 2 == 0 else "\u001b[40;37m"
        for i in range(len(uci_moves))
    ]
    colored = [
        f"{c} {i:2}: {m.strip():5}" for i, (c, m) in enumerate(zip(colors, uci_moves))
    ]
    wrapped = [txt if i % wrap_width else f"\n{txt}" for i, txt in enumerate(colored)]
    return "".join(wrapped) + "\x1b[0m"


def pgn_to_uci(pgn_string: str):
    """
    Converts a pgn string into uci.
    Example usage:
    ```
    >>> pgn_to_uci('1.e4 e5 2.Nf3 Nc6 3.Bb5')
    'e2e4 e7e5 g1f3 b8c6 f1b5'
    ```
    """
    from io import StringIO

    import chess.pgn

    pgn_io = StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    return " ".join([m.uci() for m in game.mainline_moves()])


def uci_to_board(
    uci_moves: Union[str, Iterable],
    *,
    force=False,
    fail_silent=False,
    verbose=True,
    as_board_stack=False,
    map_function: Callable = lambda x: x,
    reset_halfmove_clock=False,
) -> Union[chess.Board, List[Union[chess.Board, Any]]]:
    """Returns a chess.Board object from a string of UCI moves
    Params:
        force: If true, illegal moves are forcefully made. O/w, the rror is thrown
        verbose: Alert user via prints that illegal moves were attempted."""
    board = chess.Board()
    forced_moves = []
    did_force = False
    board_stack = [map_function(board.copy())]

    if isinstance(uci_moves, str):
        uci_moves = uci_moves.split(" ")

    for i, move in enumerate(uci_moves):
        try:
            move_obj = board.parse_uci(move)
            if reset_halfmove_clock:
                board.halfmove_clock = 0
            board.push(move_obj)
        except (chess.IllegalMoveError, chess.InvalidMoveError) as ex:
            if force:
                did_force = True
                forced_moves.append((i, move))
                piece = board.piece_at(chess.parse_square(move[:2]))
                board.set_piece_at(chess.parse_square(move[:2]), None)
                board.set_piece_at(chess.parse_square(move[2:4]), piece)
            elif fail_silent:
                if as_board_stack:
                    return board_stack
                else:
                    return map_function(board)
            else:
                if verbose:
                    print(f"Failed on (move_id, uci): ({i},{move})")
                    if as_board_stack:
                        return board_stack
                    else:
                        return map_function(board)
                else:
                    raise ex
        board_stack.append(map_function(board.copy()))
    if verbose and did_force:
        print(f"Forced (move_id, uci): {forced_moves}")

    if as_board_stack:
        return board_stack
    else:
        return map_function(board)


def tensor_to_colors(sorted_values: torch.Tensor, cm_theme="Blues") -> dict[int, str]:
    """Converts a tensor to a dictionary of hex colors according to a matplotlib colormap."""
    normalized = (sorted_values - sorted_values.min()) / (
        sorted_values.max() - sorted_values.min()
    )
    colormap = plt.colormaps[cm_theme]
    colors_list = [mcolors.to_hex(colormap(value)) for value in normalized.cpu()]
    colors_dict = {index: color for index, color in enumerate(colors_list)}
    return colors_dict


def tensor_to_svg_board(
    sorted_values: torch.Tensor,
    board: chess.Board = chess.Board(),
    cm_theme="viridis",
    **kwargs,
):
    """
    Converts a tensor of sorted values to an SVG chess board representation.

    Args:
        sorted_values (torch.Tensor): A tensor containing the sorted values for the chess board squares.
        cm_theme (str, optional): The color map theme to use. Defaults to "viridis".
        **kwargs: Additional keyword arguments to pass to the chess.svg.board function.
    """

    fill = tensor_to_colors(sorted_values, cm_theme=cm_theme)

    board = chess.svg.board(board, fill=fill, **kwargs)
    return board
