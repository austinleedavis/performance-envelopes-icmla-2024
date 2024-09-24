"""
Main question: Does to_piece + to_color = to_piece_by_coor
"""

import plotly.io as pio
import plotly_express as px
import torch
from matplotlib import animation
from plotly.graph_objs.layout._annotation import Annotation

import chess
from modeling.board_state_functions import ToColor, ToPiece, ToPieceByColor
from modeling.linear_probe import *

torch.set_grad_enabled(False)

layer = 12

to_color = MulticlassProbeModel.from_pretrained(f"models/chess-gpt2-uci-12x12x768-probes/to_color/layer-{layer:02d}/phase-0").train(False).requires_grad_(False)
to_piece = MulticlassProbeModel.from_pretrained(f"models/chess-gpt2-uci-12x12x768-probes/to_piece/layer-{layer:02d}/phase-0").train(False).requires_grad_(False)
to_piece_by_color = MulticlassProbeModel.from_pretrained(f"models/chess-gpt2-uci-12x12x768-probes/to_piece_by_color/layer-{layer}/phase-0").train(False).requires_grad_(False)

def compare(p1:MulticlassProbeModel, p2:MulticlassProbeModel, tiles = (0,0), class_id = (0,0)):
    weight1 = p1.submodules[tiles[0]].weight[class_id[0]]
    weight2 = p2.submodules[tiles[0]].weight[class_id[0]]

    return float(torch.cosine_similarity(weight1, weight2, dim=0))

compare(to_color, to_piece)
compare(to_piece_by_color, to_color)


all_comparisons = []
for square in range(64):
    pairwise_comparisons_single_square = []
    for i, color in enumerate(ToColor.class_labels):
        color_wise_sims = []
        for j, piece in enumerate(ToPiece.class_labels):
            weight = to_color.submodules[square].weight[i] + to_piece.submodules[square].weight[j]
            cos_sims = []
            for k, _ in enumerate(ToPieceByColor.class_labels):
                sim = torch.cosine_similarity(weight,to_piece_by_color.submodules[square].weight[k], dim = 0)
                cos_sims.append(sim)

            color_wise_sims.append(torch.stack(cos_sims))

        pairwise_comparisons_single_square.append(torch.stack(color_wise_sims))

    pairwise_comparisons_single_square = torch.stack(pairwise_comparisons_single_square)
    all_comparisons.append(pairwise_comparisons_single_square)

all_comparisons = torch.stack(all_comparisons)

print(f"all_comparisons.shape: {all_comparisons.shape}")
UNICODE_PIECE_SYMBOLS = {
    "R": "♖", "r": "♜",
    "N": "♘", "n": "♞",
    "B": "♗", "b": "♝",
    "Q": "♕", "q": "♛",
    "K": "♔", "k": "♚",
    "P": "♙", "p": "♟",
}

to_piece_by_color_labels = [UNICODE_PIECE_SYMBOLS[i] for i in ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']] + ['⦰']

all_comparisons.shape

class_sort_order = [0,6,1,7,2,8,3,9,4,10,5,11,12]
to_piece_by_color_labels = [to_piece_by_color_labels[i] for i in class_sort_order]


fig = px.imshow(all_comparisons[...,class_sort_order],
                facet_col = 3, animation_frame=0, 
                x=ToPiece.class_labels,
                y=ToColor.class_labels,
                labels={
                    "x":"Piece",
                    "y":"Color",
                    },
                facet_col_wrap=4,
                aspect="equal",
                zmin=0,
                zmax=1, 
                color_continuous_scale="Viridis"
                )

def update_facet_headers(annotation: Annotation):
    index = int(annotation.text.split("=")[-1])

    annotation.update(text=f"{to_piece_by_color_labels[index]}")

fig.update_layout(
    sliders=[
        dict(
            currentvalue=dict(prefix="square: "),
            steps=[dict(label=chess.SQUARE_NAMES[i]) for i in range(64)],
        )
    ]
)
fig.for_each_annotation(update_facet_headers)

fig.show()


fig = px.imshow(all_comparisons[...,class_sort_order].mean(0,keepdim=True),
                facet_col = 3,
                animation_frame=0,
                x=ToPiece.class_labels,
                y=ToColor.class_labels,
                labels={
                    "x":"Piece",
                    "y":"Color",
                    },
                facet_col_wrap=4,
                aspect="equal",
                zmin=0,
                zmax=1, 
                color_continuous_scale="Viridis"
                )

def update_facet_headers(annotation: Annotation):
    index = int(annotation.text.split("=")[-1])
    
    annotation.update(text=f"{to_piece_by_color_labels[index]}")
fig.update_layout(sliders=[{"currentvalue": {"prefix": "Square="}}])
fig.for_each_annotation(update_facet_headers)

fig.show()
# %%

to_piece_by_color_labels = [UNICODE_PIECE_SYMBOLS[i] for i in list("pPnNbBrRqQkK")] + ['⦰']

all_comparisons.shape


class_sort_order = [0,1,2,3,4,5,6,7,8,9,10,11,12]
class_sort_order = [0,1,2,3,4,5,6,7,8,9,10,11]
class_sort_order = [6,0,7,1,8,2,9,3,10,4,11,5]
class_sort_order = [0,6,1,7,2,8,3,9,4,10,5,11]

# to_piece_by_color_labels = [to_piece_by_color_labels[i] for i in class_sort_order]
x_labels= [UNICODE_PIECE_SYMBOLS[k] for k in ["P","N","B","R","Q","K"]]+["⦰"]

fig = px.imshow(
    all_comparisons[:,:2,:,class_sort_order].mean(0,keepdim=False),
    facet_col = 2,
    x=x_labels,#ToPiece.class_labels,
    y=ToColor.class_labels[:2],
    labels={
        "x":"",
        "y":"",
        },
    facet_col_wrap=2,
    facet_row_spacing=0.1,
    aspect="equal",
    zmin=-1.0,
    color_continuous_midpoint=0.0,
    zmax=1, 
    color_continuous_scale="RdBu",
    )

def update_facet_headers(annotation: Annotation):
    index = int(annotation.text.split("=")[-1])
    
    annotation.update(text=f"{to_piece_by_color_labels[index]}")

tick_font_size = 12
fig.update_xaxes(dict(title_font=dict(size=20), tickfont=dict(size=tick_font_size)))
fig.update_yaxes(dict(title_font=dict(size=20), tickfont=dict(size=tick_font_size)))
fig.update_layout(sliders=[{"currentvalue": {"prefix": "Square="}}], width=400, height=600)
fig.for_each_annotation(update_facet_headers)
for i in fig['layout']['annotations']:
    i['font'] = dict(size=20,color='#000000')

fig.for_each_xaxis(lambda ax: ax.update(showticklabels=True))

# pio.write_image(fig, 'data/icmla_paper/images/figures/Cosine Similarty By Piece.svg', format='svg')
    
fig.show()
