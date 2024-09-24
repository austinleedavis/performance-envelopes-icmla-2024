"""
The raw data metadata calls board states "clean" "dirty" and "patched".
The procesed data's metadata calls them "original" "target" and "future".
"""

# %%

import os

import chess
import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sns
import torch
from IPython.display import display
from tabulate import tabulate

torch.set_grad_enabled(False)


INTERVENTION_TYPE: str = "removal" # type of intervention
DS_CONFIG_NAME: str = "201302-moves" # dataset config
OUTPUT_FOLDER: str = 'data/icmla_paper/raw_data/raw_data' # location to save raw output data
NUM_LOGITS = 72

def compute_dfs(*, metadata, logits_clean, logits_randomized, logits_probed, logits_patched):
    logits = torch.stack([logits_clean, logits_randomized, logits_probed, logits_patched])

    # Prepare the clean/dirty masks
    board_original = [chess.Board(fen) for fen in metadata["fen_clean"]]
    moves_original = [list(set([m.from_square + 4 for m in b.legal_moves])) for b in board_original]
    mask_original = torch.zeros(len(metadata), 72, dtype=torch.bool)
    for i, token_id in enumerate(moves_original):
        mask_original[i, token_id] = True

    board_target = [chess.Board(fen) for fen in metadata["fen_dirty"]]
    moves_target = [list(set([m.from_square + 4 for m in b.legal_moves])) for b in board_target]
    mask_target = torch.zeros(len(metadata), 72, dtype=torch.bool)
    for i, token_id in enumerate(moves_target):
        mask_target[i, token_id] = True

    board_future = [chess.Board(fen) for fen in metadata["fen_patch"]]
    moves_future = [list(set([m.from_square + 4 for m in b.legal_moves])) for b in board_future]
    mask_future = torch.zeros(len(metadata), 72, dtype=torch.bool)
    for i, token_id in enumerate(moves_future):
        mask_future[i, token_id] = True

    p_mass_original = (logits*mask_original.to(torch.float16)).sum(dim=-1).T.tolist()
    p_mass_target = (logits*mask_target.to(torch.float16)).sum(dim=-1).T.tolist()
    p_mass_future = (logits*mask_future.to(torch.float16)).sum(dim=-1).T.tolist()

    df_original = pd.DataFrame(p_mass_original, columns=["None", "Randomized","Probed", "Patched"])
    df_original = df_original.melt(var_name='Intervention', value_name='Legal Move Probability Mass')
    df_original['Board State'] = 'Original'

    df_target = pd.DataFrame(p_mass_target, columns=["None",  "Randomized", "Probed", "Patched"])
    df_target = df_target.melt(var_name='Intervention', value_name='Legal Move Probability Mass')
    df_target['Board State'] = 'Target'

    df_future = pd.DataFrame(p_mass_future, columns=["None",  "Randomized", "Probed", "Patched"])
    df_future = df_future.melt(var_name='Intervention', value_name='Legal Move Probability Mass')
    df_future['Board State'] = 'Future'

    return df_original, df_target, df_future

def _load_raw_data(move_index, layer, DO_FIXES = True):
    data_filename = f"{DS_CONFIG_NAME}_{INTERVENTION_TYPE}_Mv{move_index:02}_Ly{layer:02}.h5"
    data_path = os.path.join(OUTPUT_FOLDER, data_filename)

    metadata = pd.read_hdf(data_path)

    with h5py.File(data_path, 'r') as hf:
        logits_clean = torch.tensor(np.array(hf['logits_clean']))
        logits_probed = torch.tensor(np.array(hf['logits_probed']))
        logits_randomized = torch.tensor(np.array(hf['logits_randomized']))
        logits_patched = torch.tensor(np.array(hf['logits_patched']))

    if DO_FIXES and layer == 7 and move_index == 10:
        logits_clean = torch.concat([logits_clean[:9613], logits_clean[9713:]])

    mean_cos_sim = torch.cosine_similarity(logits_clean, logits_randomized).mean()
    assert  mean_cos_sim > 0.98, f"FIXES={DO_FIXES}, CoSim={mean_cos_sim} A Cosine Similarty <0.98 may indicate bad data alignment."

    return metadata, logits_clean, logits_probed, logits_randomized, logits_patched

def _load_and_process_data(move_index, layer):
    metadata, logits_clean, logits_probed, logits_randomized, logits_patched = _load_raw_data(
        move_index, layer
    )

    df_original, df_target, df_future = compute_dfs(
        metadata=metadata,
        logits_clean=logits_clean,
        logits_randomized=logits_randomized,
        logits_probed=logits_probed,
        logits_patched=logits_patched,
    )

    return metadata, df_original, df_target, df_future

def process_raw_data():

    for layer, move_index in [(L, M) for L in range(0,12) for M in range(5,40,5)]:

        print(f"{move_index=} {layer=}")

        metadata, df_original, df_target, df_future = _load_and_process_data(move_index, layer)

        multiplier = (num_interventions := 4) * (num_dfs:=3)
        df_concat = pd.concat([df_original, df_target, df_future],ignore_index=True)

        # Save the dataframe
        df_concat["Layer"] = layer
        df_concat["Move"] = move_index
        df_concat["Piece"] = metadata["piece_type"].to_list()*multiplier
        df_concat["Square"] = metadata["target_square"].to_list()*multiplier
        df_concat["site"] = metadata["site"].to_list()*multiplier
        df_concat["fen_original"] = metadata["fen_clean"].to_list()*multiplier
        df_concat["fen_target"] = metadata["fen_dirty"].to_list()*multiplier
        df_concat["fen_future"] = metadata["fen_patch"].to_list()*multiplier

        df_filename = pkl_name(move_index, layer)
        pd.to_pickle(
            df_concat,
            df_filename,
        )


def create_violin_plot(
    *,
    df_original,
    df_target,
    df_future,
    metadata,
    move_index,
    layer,
    do_save=False,
    show_fig=True,
    suptitle=None
):

    ## %%
    # Create a figure with a specific gridspec layout
    fig = plt.figure(figsize=(8, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 4, 1])  # Adjust the width ratios as needed

    # Create the subplots using the gridspec
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)

    # Plot the violin plot for the Original Board State
    sns.violinplot(ax=ax0, data=df_original[df_original["Intervention"] == "None"], y="Legal Move Probability Mass", x="Intervention", linewidth=1, cut=0.0, fill=False)
    ax0.set_title("Original State")

    df_target["Intervention"][df_target["Intervention"] == "Patched"] = "Time Shifted"
    df_future["Intervention"][df_future["Intervention"] == "Patched"] = "Time Shifted"

    # Plot the violin plot for the Piece Removed Board State
    sns.violinplot(ax=ax1, data=df_target, y="Legal Move Probability Mass", x="Intervention", linewidth=1, cut=0.0)
    ax1.set_title("Target State")
    ax1.tick_params(axis='y', left=False, labelleft=False)
    ax1.set_ylabel('')  # Remove the y-axis label on ax1

    sns.violinplot(ax=ax2, 
                   data=df_future.query("Intervention == 'Time Shifted'"), 
                   y="Legal Move Probability Mass", x="Intervention", 
                   linewidth=1, cut=0.0, palette=sns.color_palette()[3:])

    ax2.set_title("Future State")
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.set_ylabel('')  # Remove the y-axis label on ax2

    # fig.suptitle(suptitle if suptitle else 'Comparison of Legal Move Probability Mass by Board State', fontsize=16)
    # fig.text(0.5, .89, f"(Move: {move_index}, Layer < {layer}, N={len(df_original):,})", ha='center', fontsize=12)
    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    if do_save:
        plt_filename= f"Mv{move_index:02}_Ly{layer:02}.png"
        fig.savefig(os.path.join('images',plt_filename), dpi=600, bbox_inches='tight')
    if show_fig:
        plt.show()



def create_original_board_violins(df_original):
    
    # Initialize the plot
    fig, ax = plt.subplots()

    # Get unique factors and their positions
    factors = df_original['Intervention'].unique()
    positions = range(len(factors))

    # Width of each violin plot
    width = 0.4

    # Plot each factor's violin plot
    for position, factor in zip(positions, factors):
        data = df_original[df_original['Intervention'] == factor]['Legal Move Probability Mass']
        parts = ax.violinplot(
            data,
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor([mcolors.rgb2hex(color) for color in sns.color_palette("deep")][position])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        
    # Set x-ticks to the factors
    ax.set_xticks(positions)
    ax.set_xticklabels(factors)

    return ax

def create_violin_plot2(
    *,
    df_original,
    df_target,
    df_future,
    metadata,
    move_index,
    layer,
    do_save=False,
    show_fig=True,
):
    
    colors = [mcolors.rgb2hex(color) for color in sns.color_palette("deep")]
    colors_dict = {'None':colors[1], 'Random':colors[2], 'Probe':colors[0], 'Patch':colors[3]}

    corrections = [("Randomized", "Random"),
    ("Patched", "Patch"),
    ("Probed", "Probe"),]

    for df in [df_target, df_original, df_future]:
        for (wrong, right) in corrections:
            df.loc[df['Intervention'] == wrong, 'Intervention'] = right

    
    # Create a figure with a specific gridspec layout
    scale = 0.9
    fig = plt.figure(figsize=(8*scale, 4*scale),dpi=400)
    gs = fig.add_gridspec(1, 3, width_ratios=[4, 1, 1])  # Adjust the width ratios as needed

    # Create the subplots using the gridspec
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)

    ##################

    # Get unique factors and their positions
    factors = df_original['Intervention'].unique()
    positions = range(len(factors))

    # Width of each violin plot
    width = 0.4

    # Plot each factor's violin plot
    for position, factor in zip(positions, factors):
        data = df_original[df_original['Intervention'] == factor]['Legal Move Probability Mass']
        parts = ax0.violinplot(
            data,
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        
        for pc in parts['bodies']:
            pc.set_facecolor(colors_dict[factor])
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        
    # Set x-ticks to the factors
    ax0.set_xticks(positions)
    ax0.set_xticklabels(factors)
    ax0.set_title("Original Board Positions (BP)")

    # Plot the violin plot for the Piece Removed Board State
    sns.violinplot(
        ax=ax1,
        data=df_target.query("Intervention == 'Probe'"),
        y="Legal Move Probability Mass",
        x="Intervention",
        linewidth=1,
        cut=0.0,
    )
    ax1.set_title("Target BP")
    ax1.tick_params(axis='y', left=False, labelleft=False)
    ax1.set_ylabel('')  # Remove the y-axis label on ax1
    
    sns.violinplot(ax=ax2, 
                   data=df_future.query("Intervention == 'Patch'"), 
                   y="Legal Move Probability Mass", x="Intervention", 
                   linewidth=1, cut=0.0, palette=sns.color_palette()[3:])

    ax2.set_title("Future BP")
    ax2.tick_params(axis='y', left=False, labelleft=False)
    ax2.set_ylabel('')  # Remove the y-axis label on ax2

    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    ax0.set_xlabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    if do_save:
        plt_filename= f"icmla-legal-move-probs2.svg"
        fig.savefig(os.path.join('data/icmla_paper/images/figures',plt_filename), dpi=400, bbox_inches='tight', format='svg')
    if show_fig:
        plt.show()


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

def create_statistics_tables(*dataframes):
    df_concat = pd.concat(dataframes, ignore_index=True)

    pivot_table = df_concat.pivot_table(
        values='Legal Move Probability Mass',
        index=['Intervention', 'Board State'],
        aggfunc=['mean', 'std', 'min', 'max']
    )

    # Flatten the MultiIndex columns for better readability
    pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]

    # Reset the index to convert the pivot table to a regular dataframe
    pivot_table_reset = pivot_table.reset_index()

    # Create separate tables for mean, std, min, and max
    mean_table = pivot_table_reset.pivot(index='Intervention', columns='Board State', values='mean_Legal Move Probability Mass')
    std_table = pivot_table_reset.pivot(index='Intervention', columns='Board State', values='std_Legal Move Probability Mass')
    min_table = pivot_table_reset.pivot(index='Intervention', columns='Board State', values='min_Legal Move Probability Mass')
    max_table = pivot_table_reset.pivot(index='Intervention', columns='Board State', values='max_Legal Move Probability Mass')

    # Print the resulting tables

    print("Mean Table:")
    print(tabulate(mean_table, headers='keys', tablefmt='psql'))
    print("\nStandard Deviation Table:")
    print(tabulate(std_table, headers='keys', tablefmt='psql'))
    print("\nMinimum Table:")
    print(tabulate(min_table, headers='keys', tablefmt='psql'))
    print("\nMaximum Table:")
    print(tabulate(max_table, headers='keys', tablefmt='psql'))

def save_all_violin_charts():

    metadatas = []
    df_originals = []
    df_targets = []
    df_futures = []
    
    for layer, move_index in [(L, M) for L in [7] for M in range(5,40,5)]:
        print(f"{move_index=} {layer=}")

        metadata, df_original, df_target, df_future = _load_and_process_data(move_index, layer)
        metadatas.append(metadata)
        df_originals.append(df_original)
        df_targets.append(df_target)
        df_futures.append(df_future)
    
    metadata = pd.concat(metadatas)
    df_original = pd.concat(df_originals).dropna(axis=0)
    df_target = pd.concat(df_targets).dropna(axis=0)
    df_future = pd.concat(df_futures).dropna(axis=0)


    create_violin_plot2(
        df_original=df_original,
        df_target=df_target,
        df_future=df_future,
        metadata=metadata,
        move_index=move_index,
        layer=layer,
        do_save=True,
    )
    len(metadata)

def create_ridge_plot(df: pd.DataFrame, value: str, y_factor: str):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    
    n_cat = len(df[y_factor].unique())
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(n_cat, rot=-0.25, light=0.7)
    y_factor = sns.FacetGrid(
        df,
        row=y_factor,
        hue=y_factor,
        aspect=15,
        height=0.5,
        palette=pal,
    )

    # Draw the densities in a few steps
    y_factor.map(
        sns.kdeplot, value, bw_adjust=0.5, clip_on=False, fill=True, alpha=1, linewidth=1.5
    )
    y_factor.map(sns.kdeplot, value, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    y_factor.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    y_factor.map(label, value)

    # Set the subplots to overlap
    y_factor.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    y_factor.set_titles("")
    y_factor.set(yticks=[], ylabel="")
    y_factor.despine(bottom=True, left=True)

def make_pretty_ridgeline_plot(df: pd.DataFrame, Y_FACTOR: str, VALUE: str, title_text: str, subtitle_text: str,):

    df = df[~df[VALUE].isna()]

    ngroups = df[Y_FACTOR].nunique()    # Dynamically calculate the number of rows in the chart.

    bandwidth = 1                     # Control how smooth you want the graphs

    darkgreen = "#8487C1AA"#'#9BC184'
    midgreen = "#A4B1D6AA"#'#C2D6A4'
    lowgreen = "#CBE0E7AA"#'#E7E5CB'
    colors = [lowgreen, midgreen, darkgreen, midgreen, lowgreen]

    darkgrey = '#525252'
    figsize=(8,10)

    fig, axs = plt.subplots(nrows=ngroups, ncols=1, figsize=figsize)
    axs = axs.flatten() # needed to access each individual axis

    # iterate over axes
    factors = df[Y_FACTOR].unique().tolist()
    for i, factor in enumerate(factors):
        print(f"{i}th factor")
        # subset the data for each word
        subset = df[df[Y_FACTOR] == factor]

        # plot the distribution of prices
        sns.kdeplot(
            subset[VALUE],
            fill=True,
            bw_adjust = bandwidth,
            ax=axs[i],
            color='grey',
            edgecolor='lightgrey'
        )

        # global mean reference line
        global_mean = df[VALUE].mean()
        axs[i].axvline(global_mean, color=darkgrey, linestyle='--')
        
        # # display average number of bedrooms on left
        subset_mean = subset[VALUE].mean().round(2)


        # display Row Label on left
        axs[i].text(
            -0.0265*figsize[0], # horizontal relative to whole fig
            0.5, # vertical relative to this axis
            f"{Y_FACTOR} {factor}" if factor != 0 else "Embed",
            ha='left',
            fontsize=10,
            # fontproperties=fira_sans_semibold,
            color=darkgrey
        )

        # Display Mean on left
        axs[i].text(
            -0.012*figsize[0], 
            0.5,
            f'({subset_mean})',
            ha='left',
            fontsize=10,
            # fontproperties=fira_sans_regular,
            color=darkgrey
        )

        

        ax_height = axs[i].get_ylim()[-1]

        # set title and labels
        axs[i].set_xlim(0, 1.0)
        axs[i].set_ylim(0, ax_height)
        
        axs[i].set_ylabel('')

        # compute quantiles
        quantiles = np.percentile(subset[VALUE], [2.5, 10, 25, 75, 90, 97.5])
        quantiles = quantiles.tolist()

        # fill space between each pair of quantiles
        quantiles_height =  ax_height / 8
        for j in range(len(quantiles) - 1):
            axs[i].fill_between(
                [quantiles[j], # lower bound
                quantiles[j+1]], # upper bound
                0, # max y=0
                quantiles_height, # max y=0.0002
                color=colors[j]
            )
        # median value as a reference
        median = subset[VALUE].median()
        axs[i].scatter([median], [quantiles_height/2], color='black', s=10)

        # x axis scale for last ax
        if i == ngroups-1:
            values = [0, 0.25, 0.5, 0.75, 1.0]
            for value in values:
                axs[i].text(
                    x_offset:=value, 
                    y_offset:=-5,
                    f'{value}',
                    ha='center',
                    fontsize=10
                )

        # remove axis
        axs[i].set_axis_off()

    # reference line label
    text = 'Global Mean'
    fig.text(
        global_mean, 
        y=0.88,#0.0176*figsize[1],
        s=text,
        ha='right',
        fontsize=10
    )


    # number of bedrooms label
    text = '(μ)'
    fig.text(
        0.06, 
        0.88,
        text,
        ha='left',
        fontsize=10,
        # fontproperties=fira_sans_regular,
        color=darkgrey
    )


    # x axis label
    text = "Density of Mass Reallocated to Valid Moves"
    fig.text(
        0.5, 0.03,
        text,
        ha='center',
        fontsize=14,
        # fontproperties=fira_sans_regular
    )

    # background grey lines
    from matplotlib.lines import Line2D


    def add_line(xpos, ypos, fig=fig):
        line = Line2D(
            xpos, ypos,
            color='lightgrey',
            lw=0.2,
            transform=fig.transFigure
        )
        fig.lines.append(line)
    add_line([0.317, 0.317], [0.1, 0.9])
    add_line([0.51, 0.51], [0.1, 0.9])
    add_line([0.703, 0.703], [0.1, 0.9])
    add_line([0.896, 0.896], [0.1, 0.9])

    file_title = title_text.lower().replace(" ", "_")
    plt.show()
    plt.savefig(f'data/icmla_paper/images/figures/{file_title}.png', dpi=400, bbox_inches='tight')



def make_pretty_ridgeline_plot_for_icmla(df: pd.DataFrame, Y_FACTOR: str = "Layer", VALUE: str = "Legal Move Probability Mass"):

    df = df[~df[VALUE].isna()]

    ngroups = df[Y_FACTOR].nunique()    # Dynamically calculate the number of rows in the chart.

    bandwidth = 1                     # Control how smooth you want the graphs

    darkgreen = "#8487C1AA"#'#9BC184'
    midgreen = "#A4B1D6AA"#'#C2D6A4'
    lowgreen = "#CBE0E7AA"#'#E7E5CB'
    colors = [lowgreen, midgreen, darkgreen, midgreen, lowgreen]

    darkgrey = '#000000'#'#525252'
    figsize=(4,7)

    fig, axs = plt.subplots(nrows=ngroups, ncols=1, figsize=figsize)
    axs = axs.flatten() # needed to access each individual axis

    # iterate over axes
    factors = df[Y_FACTOR].unique().tolist()
    for i, factor in enumerate(factors):
        print(f"{i}th factor")
        # subset the data for each word
        subset = df[df[Y_FACTOR] == factor]

        # plot the distribution of prices
        sns.kdeplot(
            subset[VALUE],
            fill=True,
            bw_adjust = bandwidth,
            ax=axs[i],
            color='grey',
            edgecolor='darkgrey'
        )

        # global mean reference line
        global_mean = df[VALUE].mean()
        axs[i].axvline(global_mean, color=darkgrey, linestyle='--')
        
        # # display average number of bedrooms on left
        subset_mean = subset[VALUE].mean().round(2)


        # display Row Label on left
        axs[i].text(
            -0.355, # horizontal relative to whole fig
            0.5, # vertical relative to this axis
            f"{Y_FACTOR} {factor}" if factor != 0 else "Embed",
            ha='left',
            fontsize=10,
            # fontproperties=fira_sans_semibold,
            color=darkgrey
        )

        # Display Mean on left
        axs[i].text(
            -0.155, 
            0.5,
            f'({subset_mean})',
            ha='left',
            fontsize=10,
            # fontproperties=fira_sans_regular,
            color=darkgrey
        )

        

        ax_height = axs[i].get_ylim()[-1]

        # set title and labels
        axs[i].set_xlim(0, 1.0)
        axs[i].set_ylim(0, ax_height)
        
        axs[i].set_ylabel('')

        # compute quantiles
        quantiles = np.percentile(subset[VALUE], [2.5, 10, 25, 75, 90, 97.5])
        quantiles = quantiles.tolist()

        # fill space between each pair of quantiles
        quantiles_height =  ax_height / 8
        for j in range(len(quantiles) - 1):
            axs[i].fill_between(
                [quantiles[j], # lower bound
                quantiles[j+1]], # upper bound
                0, # max y=0
                quantiles_height, # max y=0.0002
                color=colors[j]
            )

        # median value as a reference
        median = subset[VALUE].median()
        axs[i].scatter([median], [quantiles_height/2], color='black', s=10)

        # x axis scale for last ax
        if i == ngroups-1:
            values = [0, 0.25, 0.5, 0.75, 1.0]
            for value in values:
                axs[i].text(
                    x_offset:=value, 
                    y_offset:=-5,
                    f'{value}',
                    ha='center',
                    fontsize=10
                )

        # remove axis
        axs[i].set_axis_off()

    # reference line label
    text = 'Global Mean'
    fig.text(
        global_mean, 
        y=0.89,
        s=text,
        ha='right',
        fontsize=10
    )

    # mean label
    text = '(μ)'
    fig.text(
        0.06, 
        0.88,
        text,
        ha='left',
        fontsize=10,
        # fontproperties=fira_sans_regular,
        color=darkgrey
    )

    # x axis label
    text = "Density of Mass Reallocated to Valid Moves"
    fig.text(
        0.5, 0.05,
        text,
        ha='center',
        fontsize=10,
        # fontproperties=fira_sans_regular
    )

    # background grey lines
    from matplotlib.lines import Line2D


    def add_line(xpos, ypos, fig=fig):
        line = Line2D(
            xpos, ypos,
            color='lightgrey',
            lw=0.2,
            transform=fig.transFigure
        )
        fig.lines.append(line)
    add_line([0.317, 0.317], [0.1, 0.9])
    add_line([0.51, 0.51], [0.1, 0.9])
    add_line([0.703, 0.703], [0.1, 0.9])
    add_line([0.896, 0.896], [0.1, 0.9])

    # plt.show()
    plt.savefig(f'data/icmla_paper/images/figures/icmla_legal_mass_density_by_layer5.png', dpi=400, bbox_inches='tight')


def pkl_name(move: int, layer: int):
    return f"data/probe_data/Mv{move:02}_Ly{layer:02}.pkl"

# %%
# Load the data

df = pd.concat(
    [
        pd.read_pickle(f)
        for f in [
            pkl_name(M,L) for M in range(5,40,5) for L in range(13)
        ] if os.path.exists(f)
    ]
)

print(
    f"We filter the {len(df[df['Square'] == -3])} records where a non-square was the argmax.",
    f"These come from {len(df[df['Square'] == -3]['site'].unique())} unique games.",
    f"We also filtered the {len(df[df['Piece'] == '⦰'])} records where the favored piece is ⦰."
)

df = df[df['Square'] != -3]
df = df[df['Piece'] != "⦰"]
df.loc[df['Intervention'] == 'Patched', 'Intervention'] = "Time Shifted"

# %%
# This section creates heatmaps showing factor X vs factor Y

aggfuncs=['count', "mean", "std"]

# choose from: Layer, Move, Square, Piece
# For Piece vs Square, there's a better way below
x = "Square" 
y = "Layer" 
query = ''

df_subset = df.query(query) if query != '' else df

tbls = print_stats_tables(df_subset, rows=y, columns=x, aggfuncs=aggfuncs, do_print=True)

for i, agg in enumerate(aggfuncs):
    tbl = tbls[i]
    row_labels = tbl.index.values
    col_labels = tbl.columns.values

    tbl = tbl.melt(var_name=x, value_name='value')


    if "Square" in [x,y]:
        tbl["Board Row"] = tbl["Square"].map(lambda x : 0 - x // 8)

    tbl[y] = list(row_labels)*len(col_labels)

    if "Piece" in [x,y]:
        tbl["Color"] = (["White"] * 6 + ["Black"] * 6) * len(
            row_labels if x == "Piece" else col_labels
        )
        
        if y == "Piece":
            unicode_labels = [chess.UNICODE_PIECE_SYMBOLS[p] for p in row_labels]
            tbl["Piece"] = list(unicode_labels)*len(col_labels)
        else:
            unicode_labels = [chess.UNICODE_PIECE_SYMBOLS[p] for p in col_labels]
            tbl["Piece"] = list(unicode_labels)*len(row_labels)
            
    tbl['Value'] = agg # Notice capital 'V' is label, lowercase 'v' is actual value!

    tbl = tbl.dropna()
    if agg == "count":
        tbl['value'] = np.log(tbl['value'])
        tbl['Value'] = "Ln(count)" # capital 'V'!
    tbls[i] = tbl

df_agg = pd.concat(tbls, ignore_index=True)

if "Square" in [x,y]:
    df_agg["Square"] = df_agg["Square"].map(lambda x: chess.SQUARE_NAMES[x])

row_facets = "Board Row" if "Square" in [x,y] else 'Color' if "Piece" in [x,y] else None

g = sns.FacetGrid(df_agg, 
                  row=row_facets, 
                  col = "Value", 
                  height=4, 
                  aspect=1.5, 
                  sharex=False, 
                  sharey=False)
g.figure.suptitle(f'{x} v {y} "{query}"', fontsize=16)

def heatmap(data: pd.DataFrame, **kwargs):
    data_pivot = data.pivot(index=y, columns=x, values='value')
    return sns.heatmap(data_pivot,  vmin = 0.0, **kwargs)
g.map_dataframe(heatmap, cmap='viridis', annot=False, cbar=True)
g.figure.tight_layout()
g.figure.subplots_adjust(top=0.9)
plt_filename = f"probe_performance_{x}_v_{y}_with_filter_{query}.png"
g.figure.savefig(os.path.join('images',plt_filename), dpi=72, bbox_inches='tight')


# %%

aggfuncs=["std"]#, "std", "count"]

# choose from: Layer, Move, Square, Piece
# For Piece vs Square, use this
y = "Piece" 
x = "Square" 
query = 'Layer < 9 & Layer > 5 & Intervention == "Probed"'
plt_filename = f"probe_performance_square_v_piece_filter_{query}.png"

plt_title = plt_filename.replace("_"," ").replace(".png", "")

df_subset = df.query(query) if query != '' else df
df_subset=df_subset[~df_subset['Piece'].str.isupper()]
tbls = print_stats_tables(df_subset, rows=y, columns=x, aggfuncs=aggfuncs, do_print=False)


for i, agg in enumerate(aggfuncs):
    tbl = tbls[i]
    row_labels = tbl.index.values
    unicode_col_labels = [chess.UNICODE_PIECE_SYMBOLS[p] for p in row_labels]
    
    col_labels = tbl.columns.values              

    tbl = tbl.melt(var_name=x, value_name='value')
    tbl["Piece"] = list(unicode_col_labels)*len(col_labels)
    tbl["Row"] = tbl["Square"].map(lambda x: list("87654321")[(x // 8)])
    tbl["Column"] = tbl["Square"].map(lambda x: list("abcdefgh")[(x % 8)])
    tbl["Square"] = tbl["Square"].map(lambda x: chess.SQUARE_NAMES[x])
    tbl["agg_fn"] = agg
    # tbl = tbl.dropna()
    if agg == "count":
        tbl['value'] = np.log(tbl['value'])
        tbl['agg_fn'] = "log(count)"
    tbls[i] = tbl

df_agg = pd.concat(tbls)
# %%
# df_agg[df_agg['Piece'].isin(['♗', '♔', '♘', '♙', '♕', '♖'])]
g = sns.FacetGrid(df_agg, 
                  col="Piece", 
                #   row = "agg_fn", 
                  col_wrap=3,
                  
                #   row_order=list("87654321"),
                #   height=4, 
                #   aspect=1.5, 
                  sharex=True, 
                  sharey=True,)

def heatmap(data: pd.DataFrame, **kwargs):
    data_pivot = data.pivot(index="Row", columns="Column", values='value')
    subplot =  sns.heatmap(data_pivot,  
                        #    vmin = 0, vmax=1.0, 
                           square=True, 
                           linewidths = 1,
                           linecolor='white',
                           **kwargs)
    return subplot

g.map_dataframe(heatmap, cmap='RdBu', annot=False, cbar=True)

for ax in g.axes.flatten():
    ax: plt.Axes  = ax
    ax.set_xlabel('')
    ax.set_ylabel('')
    # ax.set_yticks(range(len(ax.get_yticks())))
    ax.set_yticklabels([str(int(9-label)) for label in ax.get_yticks()])
    # ax.invert_yaxis()

# cbar_ax = g.figure.add_axes([1, 0.25, 0.05, 0.5]) #for vertical orientation
cbar_ax = g.figure.add_axes([0.25, -0.01, 0.5, 0.05]) #for horizontal orientation
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
sm.set_array([])
cbar = g.figure.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label('Mean Legal Move Probability')  # Add title to the horizontal colorbar

new_headers = ['Bishop ♗', 'King ♔', 'Knight ♘', 'Pawn ♙', 'Queen ♕', 'Rook ♖']

# Set the new titles for each facet
for ax, new_header in zip(g.axes.flatten(), new_headers):
    ax.set_title(new_header)

g.figure.savefig(os.path.join('data/icmla_paper/images/figures/MLPM_by_piece_by_square.pgn'), format="svg",dpi=400, bbox_inches='tight')

# %%


make_pretty_ridgeline_plot(
    df.query("Intervention == 'Probed' & `Board State` == 'Target'"),
    Y_FACTOR = 'Layer',
    VALUE = "Legal Move Probability Mass",
    title_text = "HOW DOES LAYER DEPTH AFFECT PROBE EFFECTIVENESS?",
    subtitle_text = """
    Linear probes modify the hidden state of ChessGPT from Layer 0 to Layer k, erasing the piece from the
    model's internal world representation. By layer three, these interventions reliably guide ChessGPT to ignore
    the removed piece and reallocate probability mass among remaining valid moves.
    """,
)
