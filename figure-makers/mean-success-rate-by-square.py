# %%
import time

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

import chess

INTERVENTION_TYPE: str = "removal" # type of intervention
DS_CONFIG_NAME: str = "201302-moves" # dataset config
OUTPUT_FOLDER: str = 'data/icmla_paper/raw_data/raw_data' # location to save raw output data
NUM_LOGITS = 72

def print_stats_tables(
    df,
    rows: str,
    columns: str,
    aggfuncs: list = ["mean", "std", "min", "max"],
    values: str = "Legal Move Probability Mass",
    do_print=True
):
    pivot_table = df.pivot_table(
        values=values,
        index=[rows, columns],
        aggfunc=aggfuncs
    )

    # Flatten the MultiIndex columns for better readability
    pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]

    # Reset the index to convert the pivot table to a regular dataframe
    pivot_table_reset = pivot_table.reset_index()

    # Create separate tables for mean, std, min, and max
    agg_tables = [pivot_table_reset.pivot(index=rows, columns=columns,values = v) for v in pivot_table.columns]

    # Print the resulting tables
    if do_print:
        for agg, tbl in zip(aggfuncs,agg_tables):
            print(f"{agg.upper()}: (row,col) = ({rows}, {columns})")
            print(tabulate(tbl, headers='keys', tablefmt='psql'))

    return agg_tables

def pkl_name(move: int, layer: int):
    return f"data/probe_data/Mv{move:02}_Ly{layer:02}.pkl"

# %%
# Load the data

df = pd.concat(
    [
        pd.read_pickle(file)
        for file in [
            pkl_name(move, layer) 
                for move in range(5, 40, 5) 
                for layer in range(13)
        ]
        if os.path.exists(file)
    ]
)

df = df.query('Square != -3 and Piece !="⦰"')
print(
    f"We filter the {len(df[df['Square'] == -3])} records where a non-square was the argmax.",
    f"These come from {len(df[df['Square'] == -3]['site'].unique())} unique games.",
    f"We also filtered the {len(df[df['Piece'] == '⦰'])} records where the favored piece is ⦰."
)

# %%
df_subset = df.query('Layer > 6 & Layer < 9 & Intervention == "Probed" & `Board State` == "Target"')

## Uncomment the following line to filter on White (uppercase) or Black (lowercase)
# df_subset=df_subset[~df_subset['Piece'].str.isupper()]

aggfuncs=["mean"]#, "std", "count"]

tbls = print_stats_tables(df_subset, rows="Piece", columns="Square", aggfuncs=aggfuncs, do_print=False)

for i, agg in enumerate(aggfuncs):
    tbl = tbls[i]
    row_labels = tbl.index.values
    unicode_col_labels = [chess.UNICODE_PIECE_SYMBOLS[p] for p in row_labels]
    
    col_labels = tbl.columns.values              

    tbl = tbl.melt(value_name='value')
    tbl["Piece"] = list(unicode_col_labels)*len(col_labels)
    tbl["Row"] = tbl["Square"].map(lambda x: list("87654321")[(x // 8)])
    tbl["Column"] = tbl["Square"].map(lambda x: list("abcdefgh")[(x % 8)])
    tbl["Square"] = tbl["Square"].map(lambda x: chess.SQUARE_NAMES[x])
    tbl["agg_fn"] = agg
    # tbl = tbl.dropna()
    # if agg == "count":
    #     tbl['value'] = np.log(tbl['value'])
    #     tbl['agg_fn'] = "log(count)"
    tbls[i] = tbl

df_agg = pd.concat(tbls)

g = sns.FacetGrid(df_agg, 
                  col="Piece", 
                  col_wrap=3,
                  sharex=True, 
                  sharey=True,)

HEADERS_MAP: dict = {
    "♔": "King ♔",
    "♕": "Queen ♕",
    "♖": "Rook ♖",
    "♗": "Bishop ♗",
    "♘": "Knight ♘",
    "♙": "Pawn ♙",
    "♚": "King ♚",
    "♛": "Queen ♛",
    "♜": "Rook ♜",
    "♝": "Bishop ♝",
    "♞": "Knight ♞",
    "♟": "Pawn ♟",
}

def heatmap(data: pd.DataFrame, **kwargs):
    data_pivot = data.pivot(index="Row", columns="Column", values='value')
    subplot =  sns.heatmap(data_pivot,**kwargs)
    subplot.set_xlabel('')
    subplot.set_ylabel('')
    subplot.set_yticklabels(list("87654321"))
    piece_symbol = data['Piece'].iloc[0]
    subplot.set_title(HEADERS_MAP.get(piece_symbol,"ERR"))

    return subplot

g.map_dataframe(
    heatmap,
    cmap="RdBu",
    annot=False,
    cbar=False,
    vmin=0,
    vmax=1.0,
    square=True,
    linewidths=1,
    linecolor="white",
)

# cbar_ax = g.figure.add_axes([1, 0.25, 0.05, 0.5]) #for vertical orientation
cbar_ax = g.figure.add_axes([0.25, -0.04, 0.5, 0.05]) #for horizontal orientation
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
cbar = g.figure.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label('Mean Legal Move Probability')  # Add title to the horizontal colorbar


# %%
g.figure.savefig(f'MLPM_by_piece_by_square_{time.time()}.pgn'), format="svg",dpi=400, bbox_inches='tight')
