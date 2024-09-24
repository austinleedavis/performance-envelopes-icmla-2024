"""
This script is used to generate the raw intervention dataset(s) used in the ICMLA paper. 

"""

# %%

import argparse
import math
import os
import traceback
from dataclasses import dataclass, field, fields
from pprint import pprint
from typing import Union

import chess.svg
import datasets
import h5py
import pandas as pd
import requests
import torch
from nnsight import NNsight
from torch import Tensor, Value
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel

import chess
from modeling.board_state_functions import (BoardStateFunctionBase, ToPiece,
                                            ToPieceByColor)
from modeling.chess_utils import uci_to_board
from modeling.linear_probe import *
from modeling.uci_tokenizers import UciTileTokenizer

torch.set_grad_enabled(False)

NUM_LOGITS = 72


@dataclass(frozen=True)
class CommandLineArguments:
    # fmt: off
    INTERVENTION_TYPE: str = field(default='removal', metadata={"help":"Type of intervention. (Default='removal')"})
    INTERVENTION_MOVE_INDEX: int = field(default=10, metadata={"help":"The whole move number being intervened. (Default=10)"})
    PATCH_OFFSET: int = field(default=6, metadata={"help":"Offset must be EVEN to avoid mixing white/black states. (Default=6)"})
    ETA: float = field(default=0.5, metadata={"help":"Scale factor for the probe intervention. (Default=0.5)"})
    MAX_LAYER: int = field(default=7, metadata={"help":"Latest layer to intervene. (Default=7)"})
    FORCE_CONTINUATION_FROM_INDEX: int = field(default=-1, metadata={"help":"Allows data generation to continue from this index in the dataset. If default of -1 is used, the dataset generation will continue from where the last run ended. (Default=-1)"})
    DS_PATH: str = field(default='austindavis/lichess_uci', metadata={"help":"Dataset path. (Default='austindavis/lichess_uci')"})
    DS_CONFIG_NAME: str = field(default='201302-moves', metadata={"help":"Dataset config. (Default='201302-moves')"})
    DS_SPLIT: str = field(default='train', metadata={"help":"Dataset split. (Default='train')"})
    OUTPUT_FOLDER: str = field(default='data/icmla_paper/raw_data', metadata={"help":"Location to save raw output data. (Default='data/icmla_paper/raw_data')"})
    BATCH_SIZE: int = field(default=100, metadata={"help":"I/O batch size. (Default=100)"})
    MODEL_NAME: str = field(default='austindavis/chess-gpt2-uci-12x12x768', metadata={"help":"Name of the model to use. (Default='austindavis/chess-gpt2-uci-12x12x768')"})
    # fmt: on

    def __post_init__(self):
        if self.MAX_LAYER < 0:
            raise ValueError(f"MAX_LAYER must be non-negative, got {self.MAX_LAYER}")
        if self.INTERVENTION_TYPE != 'removal':
            raise NotImplementedError(f"The intervention type '{self.INTERVENTION_TYPE}' is not implemented.")
        if self.INTERVENTION_MOVE_INDEX < 0:
            raise ValueError(f"INTERVENTION_MOVE_INDEX must be non-negative, got {self.INTERVENTION_MOVE_INDEX}")
        if self.INTERVENTION_MOVE_INDEX + self.PATCH_OFFSET < 0:
            raise ValueError(f"INTERVENTION_MOVE_INDEX + PATCH_OFFSET must be non-negative, got {self.PATCH_OFFSET + self.INTERVENTION_MOVE_INDEX}")


@dataclass(frozen=True)
class MetaDataRecord:
    """This object stores the metadata for each record. It is written to the dataset's metadata key"""

    index: int
    site: str = field(metadata={"min_itemsize": 50})
    piece_type: str = field(metadata={"min_itemsize": 50})
    target_square: int = field(metadata={"min_itemsize": 10})
    fen_clean: str = field(metadata={"min_itemsize": 100})
    fen_dirty: str = field(metadata={"min_itemsize": 100})
    fen_patch: str = field(metadata={"min_itemsize": 100})

    @classmethod
    def get_min_itemsizes(cls):
        return {
            field.name: field.metadata["min_itemsize"]
            for field in fields(cls)
            if "min_itemsize" in field.metadata
        }
    
    @classmethod
    def get_fields_list(cls):
        return [field.name for field in fields(cls)]

def main(args: CommandLineArguments):
    # Load stuff
    state_fn = ToPieceByColor()
    tokenizer: UciTileTokenizer = UciTileTokenizer()
    model, ln_f, probes = load_models(args.MODEL_NAME)
    ds = datasets.load_dataset(
        path=args.DS_PATH, name=args.DS_CONFIG_NAME, split=args.DS_SPLIT
    )

    TOTAL_RECORDS = len(ds)
    MOVE_INDEX = args.INTERVENTION_MOVE_INDEX - 1

    output_path, continuation_index = prepare_output_files(args)

    if args.FORCE_CONTINUATION_FROM_INDEX != -1:
        continuation_index = args.FORCE_CONTINUATION_FROM_INDEX

    print(f"{output_path=}")

    # Prepare lists for caching records in batches
    metadata_records = []
    clean_records = []
    probed_records = []
    randomized_records = []
    patched_records = []

    for index in tqdm(range(continuation_index, TOTAL_RECORDS)):

        # Select the record from the dataset
        site, transcript = ds[index].values()
        uci_moves = transcript.split()
        num_moves = len(uci_moves)

        # Verify the "patch" technique will work on this sample. Skip otherwise.
        if num_moves < (MOVE_INDEX + args.PATCH_OFFSET):
            continue

        input_str, input_ids = tokenize_subgame(uci_moves, tokenizer, MOVE_INDEX)
        patch_idx = MOVE_INDEX + args.PATCH_OFFSET
        patch_str, patch_ids = tokenize_subgame(uci_moves, tokenizer, patch_idx)

        logits_clean, patch_activations = clean_pass(model, input_ids, patch_ids)

        board_clean, board_dirty, metadata = get_metadata_and_boards(
            index, site, input_str, patch_str, logits_clean
        )

        logits_probed, logits_randomized, logits_patched = compute_intervened_logits(
            model,
            ln_f,
            probes,
            state_fn,
            input_ids,
            patch_activations,
            board_clean,
            board_dirty,
            args,
        )

        metadata_records.append(vars(metadata))
        clean_records.append(logits_clean.numpy().astype("float16"))
        probed_records.append(logits_probed.numpy().astype("float16"))
        randomized_records.append(logits_randomized.numpy().astype("float16"))
        patched_records.append(logits_patched.numpy().astype("float16"))

        # Save data to disk in batches
        if len(metadata_records) >= args.BATCH_SIZE or (index + 1) >= TOTAL_RECORDS:

            save_to_disk(
                output_path,
                metadata_records,
                clean_records,
                probed_records,
                randomized_records,
                patched_records,
            )

            # Clear batch
            metadata_records.clear()
            clean_records.clear()
            probed_records.clear()
            randomized_records.clear()
            patched_records.clear()


def select_intervention_vectors(probe: MulticlassProbeModel, diff_list: list[dict]):
    """Given a list of board stae differences, computes an probe intervention vector for each layer."""
    result = torch.zeros_like(probe.submodules[0].weight[0])
    for d in diff_list:
        ids = d["ids"]
        text = d["text"]
        if text["add"] != "⦰":  # ignore empty
            result += probe.submodules[ids["square"]].weight[ids["add"]]
        if text["subtract"] != "⦰":  # ignore empty
            result -= probe.submodules[ids["square"]].weight[ids["subtract"]]

    result = result / result.norm()
    return result


def prepare_output_files(args: CommandLineArguments):
    """
    Prepares the h5py dataset. Returns the relative path to the dataset output file as 
    well as the value for the continuation index. If the dataset does not exist, the 
    continuation index is 0. If the dataset does exist, the continuation index will be 
    set so that dataset generation continues from the last unprocesed game.
    """

    OUTPUT_FILENAME = f"{args.DS_CONFIG_NAME}_{args.INTERVENTION_TYPE}_Mv{args.INTERVENTION_MOVE_INDEX:02}_Ly{args.MAX_LAYER:02}.h5"
    if not os.path.exists(args.OUTPUT_FOLDER):
        os.makedirs(args.OUTPUT_FOLDER)
    OUTPUT_PATH = os.path.join(args.OUTPUT_FOLDER, OUTPUT_FILENAME)

    if os.path.exists(OUTPUT_PATH):
        # get continuation index
        try:
            df = pd.read_hdf(OUTPUT_PATH)
        except ValueError:
            continuation_index = 0
        else:
            continuation_index = df.index.max()
            if math.isnan(continuation_index):
                continuation_index = 0
            else:
                continuation_index += 1
    else:
        # initialize the dataset
        continuation_index = 0
        metadata_columns = MetaDataRecord.get_fields_list()
        min_itemsize = MetaDataRecord.get_min_itemsizes()
        empty_df = pd.DataFrame(columns=metadata_columns)
        empty_df.set_index('index', inplace=True)
        empty_df.to_hdf(OUTPUT_PATH, key='metadata', format='table', 
                        min_itemsize=min_itemsize)
        with h5py.File(OUTPUT_PATH, "w") as hf:
            hf.create_dataset(
                "logits_clean",
                shape=(0, NUM_LOGITS),
                maxshape=(None, NUM_LOGITS),
                dtype="float16",
            )
            hf.create_dataset(
                "logits_probed",
                shape=(0, NUM_LOGITS),
                maxshape=(None, NUM_LOGITS),
                dtype="float16",
            )
            hf.create_dataset(
                "logits_randomized",
                shape=(0, NUM_LOGITS),
                maxshape=(None, NUM_LOGITS),
                dtype="float16",
            )
            hf.create_dataset(
                "logits_patched",
                shape=(0, NUM_LOGITS),
                maxshape=(None, NUM_LOGITS),
                dtype="float16",
            )
    return OUTPUT_PATH, continuation_index


def load_models(model_name: str):
    """Load and return the (model, ln_f, probes)"""
    original_model = GPT2LMHeadModel.from_pretrained(model_name).requires_grad_(False)

    model = NNsight(original_model)
    ln_f = model.transformer.ln_f  # layer norm function
    n_layers = len(model.transformer.h)

    probes = load_probes(n_layers)

    return (model, ln_f, probes)


def load_probes(n_layers: int):
    """Loads the probes for each layer into a list."""
    probes = [
        MulticlassProbeModel.from_pretrained(
            f"models/chess-gpt2-uci-12x12x768-probes/to_piece_by_color/layer-{L}/phase-0"
        )
        for L in range(n_layers + 1)
    ]
    return probes


def tokenize_subgame(
    uci_moves: list[str], tokenizer: UciTileTokenizer, move_index: int
) -> tuple[str, Tensor]:
    """
    Converts full game's uci_moves list to a subgame uci string and applies the 
    tokenizer to obtain the corresponding input_ids
    """
    input_str = " ".join(uci_moves[:move_index])
    encoding = tokenizer.batch_encode_plus(
        [input_str], return_tensors="pt", add_special_tokens=True
    )
    input_ids = encoding["input_ids"]

    return (input_str, input_ids)


def clean_pass(model: NNsight, input_ids, patch_input_ids) -> tuple[Tensor, Tensor]:
    """
    Performs a forward pass on the model. Returns (logits_clean, patch),
    where clean logits are for the non-intervention and the patch is the hidden
    state used by the patching intervention.
    """
    clean_idx = len(input_ids[0]) - 1
    patch_idx = len(patch_input_ids[0]) - 1
    with model.trace(patch_input_ids, scan=False,validate=False):
            logits_clean: Tensor = model.lm_head.output[0, clean_idx].softmax(-1).save()
            patch_activations = torch.stack(
                [h.output[0][0, patch_idx] for h in model.transformer.h]
            ).save()
    return logits_clean, patch_activations


def get_metadata_and_boards(index, site, input_str, patch_input_str, logits_clean):
    """
    Used to obtain the metadata record and clean/dirty boards.
    Returns (board_clean, board_dirty, metadata)
    """
    board_patch = uci_to_board(patch_input_str.lower())
    board_clean = uci_to_board(input_str.lower())
    board_dirty = board_clean.copy(stack=False)
    target_square = int(logits_clean.argmax(-1)) - 4
    piece_type = board_dirty.remove_piece_at(target_square)
    piece_symbol = piece_type.symbol() if piece_type is not None else "⦰"

    metadata = MetaDataRecord(
        index=index,
        site=site,
        piece_type=piece_symbol,
        target_square=target_square,
        fen_clean=board_clean.fen(),
        fen_dirty=board_dirty.fen(),
        fen_patch=board_patch.fen(),
    )
    return (board_clean, board_dirty, metadata)


def compute_intervened_logits(
    model: NNsight,
    ln_f,
    probes: list[MulticlassProbeModel],
    state_fn: BoardStateFunctionBase,
    input_ids,
    patch_activations: Tensor,
    board_clean: chess.Board,
    board_dirty: chess.Board,
    args: CommandLineArguments,
):
    """
    Performs all three intervention techniques.
    Returns: (logits_probed, logits_randomized, logits_patched)
    """

    pos = [(input_ids.shape[-1]) - 1]  # token position to intervene

    # Perform the interventions
    with model.trace(input_ids, validate=False, scan=False):

        # Perform Probe Intervention
        differences = state_fn.diff(board_clean, board_dirty)
        probe_interventions = [
            select_intervention_vectors(probe, differences) for probe in probes
        ]
        intervene_single_layer(
            probe_interventions, model, ln_f, pos, args.MAX_LAYER, args.ETA
        )
        logits_probed: Tensor = model.lm_head.output.detach().softmax(-1)[0, -1].save()

    with model.trace(input_ids, validate=False, scan=False):

        # Perform Randomized Intervention
        randomized_interventions = torch.rand_like(patch_activations)
        randomized_interventions = (
            randomized_interventions / randomized_interventions.norm()
        )
        intervene_single_layer(
            randomized_interventions, model, ln_f, pos, args.MAX_LAYER, args.ETA
        )
        logits_randomized: Tensor = (
            model.lm_head.output.detach().softmax(-1)[0, -1].save()
        )
        
    with model.trace(input_ids, validate=False, scan=False):
        # Perform Patched Intervention

        intervene_single_layer(
            patch_activations, model, ln_f, pos, args.MAX_LAYER, args.ETA
        )
        logits_patched: Tensor = model.lm_head.output.detach().softmax(-1)[0, -1].save()

    return (logits_probed, logits_randomized, logits_patched)


def save_to_disk(
    output_path: str,
    metadata_records: list[MetaDataRecord],
    clean_records: list[Tensor],
    probed_records: list[Tensor],
    randomized_records: list[Tensor],
    patched_records: list[Tensor],
):
    """
    Saves all the records to the hd5 dataset
    """

    num_new_records = len(metadata_records)

    # Save metadata to HDF5 dataset on disk
    df = pd.DataFrame(metadata_records)
    df.set_index("index", inplace=True)
    min_itemsize = MetaDataRecord.get_min_itemsizes()
    df.to_hdf(
        output_path,
        key="metadata",
        mode="a",
        append=True,
        format="table",
        min_itemsize=min_itemsize
    )

    # Write logits to HDF5 datasets on disk
    with h5py.File(output_path, "a") as hf:
        # fmt: off
        hf["logits_clean"].resize((hf["logits_clean"].shape[0] + num_new_records, NUM_LOGITS))
        hf["logits_clean"][-num_new_records:] = clean_records

        hf["logits_probed"].resize((hf["logits_probed"].shape[0] + num_new_records, NUM_LOGITS))
        hf["logits_probed"][-num_new_records:] = probed_records

        hf["logits_randomized"].resize((hf["logits_randomized"].shape[0] + num_new_records, NUM_LOGITS))
        hf["logits_randomized"][-num_new_records:] = randomized_records

        hf["logits_patched"].resize((hf["logits_patched"].shape[0] + num_new_records, NUM_LOGITS))
        hf["logits_patched"][-num_new_records:] = patched_records
        # fmt: on


def intervene(
    intervention_stack: Union[Tensor, list[Tensor]],
    model: NNsight,
    ln_f,
    pos: int,
    MAX_LAYER: int,
    ETA: float,
):
    """
    Performs an intervention by adding the intervention vectors to each layer of the 
    model's hidden state during a forward pass.

    This must be accomplished within an NNSight trace intervention. 
    """
    for layer, intervention in enumerate(intervention_stack):
        if layer > MAX_LAYER:
            continue
        initial_magnitude = ln_f(model.transformer.h[layer].output[0][0, pos]).norm()
        model.transformer.h[layer].output[0][0, pos] += (
            ETA * initial_magnitude * intervention
        )


def intervene_single_layer(
    intervention_stack: Union[Tensor, list[Tensor]],
    model: NNsight,
    ln_f,
    pos: int,
    layer: int,
    ETA: float,
):
    """
    Performs an intervention by adding the intervention vectors to each layer of the 
    model's hidden state during a forward pass.

    This must be accomplished within an NNSight trace intervention. 
    """
    intervention = intervention_stack[layer]
    initial_magnitude = ln_f(model.transformer.h[layer].output[0][0, pos]).norm()
    model.transformer.h[layer].output[0][0, pos] += (
        ETA * initial_magnitude * intervention
    )

def select_intervention_vectors(probe: MulticlassProbeModel, diff_list: list[dict]):
    """
    Selects the probe's intervention vectors for the given intervention.
    """
    result = torch.zeros_like(probe.submodules[0].weight[0])
    for d in diff_list:
        ids = d["ids"]
        text = d["text"]
        if text["add"] != "⦰":  # ignore empty
            result += probe.submodules[ids["square"]].weight[ids["add"]]
        if text["subtract"] != "⦰":  # ignore empty
            result -= probe.submodules[ids["square"]].weight[ids["subtract"]]

    result = result / result.norm()
    return result

def create_argparser():
    parser = argparse.ArgumentParser(
        description="Process command line arguments for chess intervention"
    )
    for f in fields(CommandLineArguments):
        help_text = f.metadata.get("help", "")
        parser.add_argument(
            f"--{f.name}", type=type(f.default), default=f.default, help=help_text
        )
    return parser


def send_error_notification(error_message, args):
    url = "https://ntfy.sh/awesomesauceisinteresting"
    newline = '\n'
    data = f"""Error occurred: {error_message}
    CommandLineArguments: {newline}{newline.join([f'    --{k}={v}' for (k,v) in vars(args).items()])}"""
    response = requests.post(url, data=data)
    print(f"Notification sent with status code: {response.status_code}")


# %%

if __name__ == "__main__":

    parser = create_argparser()
    parsed_args = parser.parse_args()
    args = CommandLineArguments(**vars(parsed_args))

    print("CLI arguments:")
    pprint(vars(args))

    try:
        print("Dataset Generation Started")
        main(args)
    except Exception as e:
        error_message = e #traceback.format_exc()
        send_error_notification(error_message, args)
        raise e
