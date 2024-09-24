# %%

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


# %%
# Load the data
def pkl_name(move: int, layer: int):
    return f"data/probe_data/Mv{move:02}_Ly{layer:02}.pkl"


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

# %%

def make_pretty_ridgeline_plot_for_icmla(df: pd.DataFrame, Y_FACTOR: str = "Layer", VALUE: str = "Legal Move Probability Mass"):

    df = df[~df[VALUE].isna()]

    ngroups = df[Y_FACTOR].nunique()    # Dynamically calculate the number of rows in the chart.

    bandwidth = 1   # Control how smooth you want the graphs

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
    for i, factor in enumerate(tqdm(factors)):
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
    def add_line(xpos, ypos, fig=fig):
        line = Line2D(
            xpos, ypos,
            color='#bbbbbbdd',
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



make_pretty_ridgeline_plot_for_icmla(
    df.query("Intervention == 'Probed' & `Board State` == 'Target'"),
)