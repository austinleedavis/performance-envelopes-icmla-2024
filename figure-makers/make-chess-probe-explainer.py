import chess
import chess.svg
import regex as re
import seaborn as sns
from numpy import empty


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return int(val)


def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip("#")

    if scalefactor < 0 or len(hexstr) < 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16)
    if len(hexstr) == 8:
        a = hexstr[-2:]

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return ("#%02x%02x%02x" % (r, g, b)) + a

def get_html_tail(x_off, y_off):
    tail = f"""
                            
        </div>

        <script>
            const offsetX = {x_off}; // Fixed x offset
            const offsetY = {y_off}; // Fixed y offset
        """
    tail += """

            const svgs = document.querySelectorAll('.svg-layer');
            svgs.forEach((svg, index) => {
                svg.style.transform = `translate(${300-index * offsetX}px, ${index * offsetY}px)`;
            });
        </script>
    </body>
    </html>

        """
    return tail

def get_html_header(probe_target: str):

    header = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Overlay SVGs</title>
        <style>
            .svg-container {
                position: relative;
                width: 400px; /* Adjust width and height as needed */
                height: 400px;
                stroke-opacity: 0.9;
            }
            .svg-layer {
                position: absolute;
                top: 0;
                left: 0;
            }
            ."""
    header += probe_target

    header += """ {
            stroke: white;
            stroke-width: 3.5px;
            stroke-opacity: 90%;
            opacity: 100%;
            outline: 3.5px solid black;
            /*
            border-radius: 5px;
            transform: translateX(300px);
            */
        }
                    
        .square:not(.h8) {
            opacity: 0.8;
            stroke: black;
            stroke-opacity: 0.0;
            stroke-width: 1px;
        }
                    
        .border {
            stroke: black;
            stroke-opacity: 100%:
            stroke-width: 1px;
        }

        
     
        </style>
    </head>
        """
    return header

def main(
    probe_target: str,
    fill_alpha: float,
    square_alpha: float,
    x_off: float,
    y_off: float,
    output_file: str
):
    probe_target_id = chess.SQUARE_NAMES.index(probe_target)

    fill_alpha_hex = f"{f'{int(fill_alpha*256):x}':02}"
    square_alpha_hex = "00"  # f"{f'{int(square_alpha*256):x}':02}"

    palette = sns.color_palette("Paired", n_colors=12).as_hex()
    palette.append("#eeeeee")
    palette = (palette)
    palette = [c + fill_alpha_hex for c in palette]

    chess.svg.board(
        board,
        fill={
            sq: palette[2 * piece.piece_type - (1 if board.color_at(sq) else 2)]
            for (sq, piece) in board.piece_map().items()
        },
        colors={
            "square light": "#eeeeee" ,
            "square dark": "#dddddd",
        },
        
        borders=True,
        coordinates=True,
        # squares=[chess.B1],

    )

    html_doc_string = get_html_header(probe_target)

    html_doc_string += """<body><div class="svg-container">"""

    # Setup piece_type/color
    empties = [(None, False)]
    occupies = [
        (color, piece_type)
        for piece_type in reversed(chess.PIECE_TYPES)
        for color in reversed(chess.COLORS)
    ]

    for index, (color, piece_type) in enumerate(empties + occupies):
        print(f"{index=} {color=} {piece_type=}")

        single_piece_board = board.copy()
        single_piece_board.set_piece_map(
            {
                sq: piece
                for (sq, piece) in board.piece_map().items()
                if piece.piece_type == piece_type
                if board.color_at(sq) == color
            }
        )
        # if len(single_piece_board.piece_map().items()) == 0:
        #     continue

        target_fill = f"{palette[index][:-2]}ff"
        # fills[probe_target_id] = target_fill

        light_color = palette[index]

        dark_color = colorscale(light_color, 0.8)

        svg = chess.svg.board(
            single_piece_board,
            # fill=fills,
            colors={"square light": light_color, "square dark": dark_color},
            borders=True,
            coordinates=False,
        )
        # Adjust the viewbox to ensure borders aren't clipped
        svg = svg.replace('viewBox="0 0 362 362"', 'viewBox="-10 -10 700 700"')

        # force add class so we can position aboslute
        svg = svg.replace("<svg ", '<svg class="svg-layer" ')

        # Fix stroke & fill on the target square
        svg = svg.replace(
            f'{probe_target}" stroke="none" fill="#ffffff" ',
            f'{probe_target}" stroke="{target_fill}" fill="{target_fill[:-2]}8b" ',
        )
        svg = svg.replace(
            f'{probe_target}" stroke="none" fill="#dddddd" ',
            f'{probe_target}" stroke="{target_fill}" fill="{target_fill[:-2]}8b" ',
        )

        # Fix board border color to match piece type
        svg = svg.replace(
            f'''<rect x="0.5" y="0.5" width="361" height="361" fill="none" stroke="#111"''',
            f'''<rect class="border" x="0.5" y="0.5" width="361" height="361" fill="none" outline="1.5px solid {target_fill}"''',
        )

        html_doc_string += svg

    html_doc_string += get_html_tail(x_off, y_off)

    html_doc_string = html_doc_string.replace("<rect", "\n            <rect")

    with open(output_file, mode="w") as f:
        f.write(html_doc_string)


if __name__ == "__main__":
    output_file1 = "images/figures/board-probes.htm"
    output_file2 = "images/figures/board-probes2.htm"

    probe_target = "h8"
    fill_alpha = 0.70
    square_alpha = 0.01
    x_off = 23.5
    y_off = 13
    board = chess.Board(
        # "rn1qkb1r/1p3ppp/p2pbn2/4p3/4P3/1NN1BP2/PPP3PP/R2QKB1R b KQkq - 0 4"
        # "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 1"
        # "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 1"
        "rnbqk1nr/2p1ppbp/p2p2p1/1p6/3PP3/2N1BP1N/PPPQ2PP/R3KB1R b KQkq - 0 4"
    )

    main(
        probe_target,
        fill_alpha,
        square_alpha,
        x_off,
        y_off,
        output_file1,
    )
