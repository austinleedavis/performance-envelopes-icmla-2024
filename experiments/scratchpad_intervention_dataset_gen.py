# %%

import argparse
import os

import chess.svg
import datasets
import h5py
import pandas as pd
import plotly_express as px
import torch
from IPython.display import display
from nnsight import NNsight
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

import chess
from modeling.board_state_functions import ToPiece, ToPieceByColor
from modeling.chess_utils import uci_to_board
from modeling.linear_probe import *
from modeling.uci_tokenizers import UciTileTokenizer

torch.set_grad_enabled(False)
# %%
# PARAMETERS
INTERVENTION_TYPE: str = "removal" # type of intervention
INTERVENTION_MOVE_INDEX: int = 10 # the whole move number being intervened
PATCH_OFFSET: int = 6 # offset must be EVEN to avoid mixing white/black states
ETA: float = 0.5 # scale factor for the probe intervention
MAX_LAYER: int = 7 # latest layer to intervene
DS_START_INDEX: int = 0 
DS_PATH: str = 'austindavis/lichess_uci' # dataset path
DS_CONFIG_NAME: str = "201302-moves" # dataset config
DS_SPLIT: str = "train" # dataset split
OUTPUT_FOLDER: str = 'data/icmla_paper/raw_data' # location to save raw output data
BATCH_SIZE: int = 100  # I/O batch size
MODEL_NAME = 'austindavis/chess-gpt2-uci-12x12x768'

MOVE_INDEX = INTERVENTION_MOVE_INDEX - 1

OUTPUT_FILENAME = f"{DS_CONFIG_NAME}_{INTERVENTION_TYPE}_Mv{INTERVENTION_MOVE_INDEX:02}_Ly{MAX_LAYER:02}.h5"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
NUM_LOGITS = 72


# %%
tokenizer: UciTileTokenizer = UciTileTokenizer()

original_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).requires_grad_(False)

model = NNsight(original_model)
ln_f = model.transformer.ln_f # layer norm function
N_LAYERS = len(model.transformer.h)
D_MODEL = model

print(model)

state_fn = ToPieceByColor()

probes = [MulticlassProbeModel.from_pretrained(f"models/chess-gpt2-uci-12x12x768-probes/to_piece_by_color/layer-{TARGET_LAYER}/phase-0")
          for TARGET_LAYER in range(N_LAYERS+1)]

ds = datasets.load_dataset(path=DS_PATH, name=DS_CONFIG_NAME, split=DS_SPLIT)
print(ds)
TOTAL_RECORDS = len(ds)

# %%


def plot_board_diffs(clean:chess.Board, dirty: chess.Board, differences):
    squares = [d['ids']['square']
         for d in differences]
    
    checkerboard = ["#fff" if (r+c)%2 else "#ddd" for r in range(8) for c in range(8)]
    clean_fill = {
        id: (
            "red"
            if (id in squares and clean.piece_at(id) is not None)
            else checkerboard[id]
        )
        for id in range(64)
    }
    dirty_fill = {
        id: (
            "lime"
            if (id in squares and dirty.piece_at(id) is not None)
            else checkerboard[id]
        )
        for id in range(64)
    }
    clean_svg =  chess.svg.board(clean,fill=clean_fill, size=500)
    dirty_svg =  chess.svg.board(dirty, fill=dirty_fill, size=500)
    display("Clean:",clean_svg,"Dirty",dirty_svg)


def compute_steering_vector(probe: MulticlassProbeModel, diff_list: list[dict]):

    result = torch.zeros_like(probe.submodules[0].weight[0])
    for d in diff_list:
        ids = d["ids"] 
        text = d['text']
        if text['add'] != '⦰': # ignore empty
            result += probe.submodules[ids["square"]].weight[ids["add"]]
        if text['subtract'] != '⦰': # ignore empty
            result -= probe.submodules[ids["square"]].weight[ids["subtract"]]

    result = result / result.norm()
    return result


def intervene(pos, intervention_stack):
    for layer, intervention in enumerate(intervention_stack):
        if layer > MAX_LAYER:
            continue
        initial_magnitude = ln_f(model.transformer.h[layer].output[0][0,pos]).norm()
        model.transformer.h[layer].output[0][0,pos] += ETA*initial_magnitude*intervention

# %%
### Only create once!
# with h5py.File(OUTPUT_PATH, 'w') as hf:
#     hf.create_dataset('logits_clean', shape=(0, NUM_LOGITS), maxshape=(None, NUM_LOGITS), dtype='float16')
#     hf.create_dataset('logits_probed', shape=(0, NUM_LOGITS), maxshape=(None, NUM_LOGITS), dtype='float16')
#     hf.create_dataset('logits_randomized', shape=(0, NUM_LOGITS), maxshape=(None, NUM_LOGITS), dtype='float16')
#     hf.create_dataset('logits_patched', shape=(0, NUM_LOGITS), maxshape=(None, NUM_LOGITS), dtype='float16')

# %%

metadata_records = []
clean_records = []
probed_records = []
randomized_records = []
patched_records = []
# %%

for index in tqdm(range(65_534, TOTAL_RECORDS)):

    # Select the record from the dataset
    record = ds[index]
    site, transcript = record.values()
    uci_moves = transcript.split()
    num_moves = len(uci_moves)

    # Verify the "patch" technique will work on this sample. Skip otherwise.
    if MOVE_INDEX+PATCH_OFFSET > num_moves:
        continue

    # Prepare input strings for the forward pass
    input_str = " ".join(uci_moves[:MOVE_INDEX])
    encoding = tokenizer.batch_encode_plus([input_str], return_tensors='pt',add_special_tokens=True)
    input_ids = encoding['input_ids']

    patch_input_str = " ".join(uci_moves[:MOVE_INDEX+PATCH_OFFSET])
    patch_encoding = tokenizer.batch_encode_plus([patch_input_str], return_tensors='pt',add_special_tokens=True)
    patch_input_ids = patch_encoding['input_ids']

    # Save clean logits and patch activations during forward pass
    clean_idx = len(input_ids[0]) - 1 
    patch_idx = len(patch_input_ids[0]) - 1 
    with model.trace() as tracer:
        with tracer.invoke(patch_input_ids) as invoker:
            logits_clean: torch.Tensor = model.lm_head.output[0,clean_idx].softmax(-1).save()
            patch = torch.stack([h.output[0][0,patch_idx] for h in model.transformer.h]).save()

    ## Create Clean, Patch, and Dirty Board States
    board_patch = uci_to_board(patch_input_str.lower())
    board_clean = uci_to_board(input_str.lower())
    board_dirty = board_clean.copy(stack=False)
    target_square = int(logits_clean.argmax(-1))-4
    piece_type = board_dirty.remove_piece_at(target_square)
    piece_symbol = piece_type.symbol() if piece_type is not None else "⦰"

    # Prepare Probe intervention
    differences = state_fn.diff(board_clean,board_dirty)
    probe_interventions = [compute_steering_vector(probe, differences) for probe in probes]

    pos = [(input_ids.shape[-1])-1] # token position to intervene

    with model.trace() as tracer:

        # Perform Probe Intervention
        with tracer.invoke(input_ids) as invoker:
            for layer, intervention in enumerate(probe_interventions):
                if layer > MAX_LAYER:
                    continue
                initial_magnitude = ln_f(model.transformer.h[layer].output[0][0,pos]).norm()
                model.transformer.h[layer].output[0][0,pos] += ETA*initial_magnitude*intervention
            logits_probed: torch.Tensor = model.lm_head.output.detach().softmax(-1)[0, -1].save()

        # Perform Randomized Intervention
        with tracer.invoke(input_ids) as invoker:
            randomized_interventions = torch.rand_like(patch)
            randomized_interventions = randomized_interventions / randomized_interventions.norm()
            intervene(pos, randomized_interventions)
            logits_randomized: torch.Tensor = model.lm_head.output.detach().softmax(-1)[0, -1].save()

        # Perform Patched Intervention
        with tracer.invoke(input_ids) as invoker:
            intervene(pos, patch)
            logits_patched: torch.Tensor = model.lm_head.output.detach().softmax(-1)[0, -1].save()

    # Cache data from this game
    metadata = dict(
        index=index,
        site=site,
        piece_type=piece_symbol,
        target_square=target_square,
        fen_clean=board_clean.fen(),
        fen_dirty=board_dirty.fen(),
        fen_patch=board_patch.fen(),
    )
    metadata_records.append(metadata)
    clean_records.append(logits_clean.numpy().astype('float16'))
    probed_records.append(logits_probed.numpy().astype('float16'))
    randomized_records.append(logits_randomized.numpy().astype('float16'))
    patched_records.append(logits_patched.numpy().astype('float16'))

    # Save data to disk in batches
    if len(metadata_records) >= BATCH_SIZE or (index + 1) >= TOTAL_RECORDS:

        num_new_records = len(metadata_records)

        # Save metadata to HDF5 dataset on disk
        df = pd.DataFrame(metadata_records)
        df.set_index('index', inplace=True)
        df.to_hdf(OUTPUT_PATH, key="metadata", mode="a", append=True)

        # Write logits to HDF5 datasets on disk
        with h5py.File(OUTPUT_PATH, 'a') as hf:
            hf['logits_clean'].resize((hf['logits_clean'].shape[0] + num_new_records, NUM_LOGITS))
            hf['logits_clean'][-num_new_records:] = clean_records

            hf['logits_probed'].resize((hf['logits_probed'].shape[0] + num_new_records, NUM_LOGITS))
            hf['logits_probed'][-num_new_records:] = probed_records

            hf['logits_randomized'].resize((hf['logits_randomized'].shape[0] + num_new_records, NUM_LOGITS))
            hf['logits_randomized'][-num_new_records:] = randomized_records

            hf['logits_patched'].resize((hf['logits_patched'].shape[0] + num_new_records, NUM_LOGITS))
            hf['logits_patched'][-num_new_records:] = patched_records

        # Clear batch
        metadata_records = []
        clean_records = []
        probed_records = []
        randomized_records = []
        patched_records = []


def main(
    *,
    INTERVENTION_TYPE,
    INTERVENTION_MOVE_INDEX,
    PATCH_OFFSET,
    ETA,
    MAX_LAYER,
    DS_PATH,
    DS_CONFIG_NAME,
    DS_SPLIT,
    OUTPUT_FOLDER,
    BATCH_SIZE,
):
    pass
    MOVE_INDEX = INTERVENTION_MOVE_INDEX - 1

    OUTPUT_FILENAME = f"{DS_CONFIG_NAME}_{INTERVENTION_TYPE}_Mv{INTERVENTION_MOVE_INDEX:02}_Ly{MAX_LAYER:02}.h5"
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    NUM_LOGITS = 72

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for performing interventions in transformer models on chess UCI move sequences.")

    parser.add_argument('--INTERVENTION_TYPE', type=str, default="removal", help='Type of intervention')
    parser.add_argument('--INTERVENTION_MOVE_INDEX', type=int, default=10, help='The whole move number being intervened')
    parser.add_argument('--PATCH_OFFSET', type=int, default=6, help='Offset must be EVEN to avoid mixing white/black states')
    parser.add_argument('--ETA', type=float, default=0.5, help='Scale factor for the probe intervention')
    parser.add_argument('--MAX_LAYER', type=int, default=7, help='Latest layer to intervene')
    parser.add_argument('--DS_PATH', type=str, default='austindavis/lichess_uci', help='Dataset path')
    parser.add_argument('--DS_CONFIG_NAME', type=str, default="201302-moves", help='Dataset config')
    parser.add_argument('--DS_SPLIT', type=str, default="train", help='Dataset split')
    parser.add_argument('--OUTPUT_FOLDER', type=str, default='data/icmla_paper/raw_data', help='Location to save raw output data')
    parser.add_argument('--BATCH_SIZE', type=int, default=100, help='I/O batch size')
    parser.add_argument('--MODEL_NAME', type=str, default='austindavis/chess-gpt2-uci-12x12x768', help='Name of the model to use')

    args = parser.parse_args("")
    main(**vars(args))


    


####################
#### Display Results
####################

# plot_board_diffs(board_clean, board_dirty, differences)
# display(px.imshow(logits_clean[range(4,68)].view(8,8).flip(0), x=list("abcdefgh"), y=list("87654321"), title="clean"))
# display(px.imshow(logits_probed[range(4,68)].view(8,8).flip(0), x=list("abcdefgh"), y=list("87654321"), title="probed"))
# display(px.imshow(logits_randomized[range(4,68)].view(8,8).flip(0), x=list("abcdefgh"), y=list("87654321"), title="randomized"))
# display(px.imshow(logits_patched[range(4,68)].view(8,8).flip(0), x=list("abcdefgh"), y=list("87654321"), title="patched"))
# board_patch


# %%

########
## Restructure metadata table
########

# existing_df = pd.read_hdf(OUTPUT_PATH, key='metadata')

# temp_df = existing_df.copy()
# with pd.HDFStore(OUTPUT_PATH, mode='a') as store:
#     del store['metadata']

# temp_df.to_hdf(
#     OUTPUT_PATH,
#     key="metadata",
#     mode="a",
#     format="table",
#     min_itemsize={
#         "site": 50,
#         "piece_type": 50,
#         "target_square": 10,
#         "fen_clean": 100,
#         "fen_dirty": 100,
#         "fen_patch": 100,
#     },
# )

# %%

#######################
#### USAGE EXAMPLE
#######################

ANALYSIS_BATCH_SIZE = 2000

df = pd.read_hdf(OUTPUT_PATH)

with h5py.File(OUTPUT_PATH, 'r') as hf:
    logits_clean = torch.tensor(hf['logits_clean'][-ANALYSIS_BATCH_SIZE:])
    logits_probed = torch.tensor(hf['logits_probed'][-ANALYSIS_BATCH_SIZE:])
    logits_randomized = torch.tensor(hf['logits_randomized'][-ANALYSIS_BATCH_SIZE:])
    logits_patched = torch.tensor(hf['logits_patched'][-ANALYSIS_BATCH_SIZE:])


# %%

#####
## How much probability mass is associated with legal moves?
#####

df_subset = df.iloc[-ANALYSIS_BATCH_SIZE:]

# Prepare the clean/dirty masks
board_clean = [chess.Board(fen) for fen in df_subset["fen_clean"]]
moves_clean = [list(set([m.from_square + 4 for m in b.legal_moves])) for b in board_clean]
mask_clean = torch.zeros(ANALYSIS_BATCH_SIZE, 72, dtype=torch.bool)
for i, token_id in enumerate(moves_clean):
    mask_clean[i, token_id] = True

board_dirty = [chess.Board(fen) for fen in df_subset["fen_dirty"]]
moves_dirty = [list(set([m.from_square + 4 for m in b.legal_moves])) for b in board_dirty]
mask_dirty = torch.zeros(ANALYSIS_BATCH_SIZE, 72, dtype=torch.bool)
for i, token_id in enumerate(moves_dirty):
    mask_dirty[i, token_id] = True


logits = torch.stack([logits_clean, logits_probed, logits_randomized, logits_patched])

p_mass_clean = (logits*mask_clean.float()).sum(dim=-1).T.tolist()
p_mass_dirty = (logits*mask_dirty.float()).sum(dim=-1).T.tolist()

df_clean = pd.DataFrame(p_mass_clean, columns=["None", "Probed", "Randomized", "Patched"])
df_clean = df_clean.melt(var_name='Intervention', value_name='Legal Move Probability Mass')
df_clean['Board State'] = 'Original'
df_dirty = pd.DataFrame(p_mass_dirty, columns=["None", "Probed", "Randomized", "Patched"])
df_dirty = df_dirty.melt(var_name='Intervention', value_name='Legal Move Probability Mass')
df_dirty['Board State'] = 'Piece Removed'
df_concat = pd.concat([df_clean,df_dirty],ignore_index=True)
px.box(
    df_concat,
    x="Intervention",
    y="Legal Move Probability Mass",
    color="Board State",
    title="Probability Mass Assigned to Legal Start Squares (Move No. 10)",
)


px.violin(
    df_concat,
    facet_col="Intervention",
    y="Legal Move Probability Mass",
    color="Board State",
    # marginal="histogram",
    title="Cumulative Density Function for interventions",
    points=False
    # lines=False,
    # markers=False,
)


# %%
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
import tabulate

print("Mean Table:")
print(tabulate(mean_table, headers='keys', tablefmt='psql'))
print("\nStandard Deviation Table:")
print(tabulate(std_table, headers='keys', tablefmt='psql'))
print("\nMinimum Table:")
print(tabulate(min_table, headers='keys', tablefmt='psql'))
print("\nMaximum Table:")
print(tabulate(max_table, headers='keys', tablefmt='psql'))
