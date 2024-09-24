"""
Creates probe training datasets. Output is saved as a csv in the folder `activations/L#/P#` where L and P represent the phase.


Example Usage to generate an activation dataset:
```
python dataset_generation/prepare_hidden_states.py record_activations data/activations-768 201301-moves austindavis/lichess_uci 
train austindavis/chess-gpt2-uci-12x12x768
```

Example usage to push local activation dataset to huggingface:

```
for P in (seq 1 4)
    for L in (seq 0 12)
        echo Layer $L Phase $P;
        python dataset_generation/prepare_hidden_states.py push_to_hub data/activations-768 austindavis/chess-gpt2-hiddenstates-768 -l $L -p $P;
    end
end
```

"""


import argparse
import os
from dataclasses import dataclass
from io import BufferedWriter
from typing import List, Tuple, Union

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

import chess
from dataset_generation.command_pattern import AbstractCommand, CommandExecutor
from modeling.chess_utils import uci_to_board
from modeling.uci_tokenizers import UciTileTokenizer

torch._C._set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser()
    executor = CommandExecutor(
        {"record_activations":ActivationDatasetGenerator(),
         "push_to_hub":HubPusher()})

    parser = executor.add_commands_to_argparser(parser)

    args = parser.parse_args()

    executor.execute_from_args(args, cfg=args)

class HubPusher(AbstractCommand):
    """Pushes hidden state vectors for given layer and phase to the Huggingface 🤗 Hub"""

    split_name = "train"


    def add_arguments(self, parser):
        # fmt: off
        parser.add_argument("data_dir", type=str,help="Directory where processed files are saved") 
        parser.add_argument("ds_repo", type=str, help="Hf 🤗 repository to which dataset will be published") 
        parser.add_argument("-l","--layer", type=int, required=False, help="The layer to process")
        parser.add_argument("-p","--phase", type=int, required=False, help="The phase to process")
        # fmt: on
        return parser

    def execute(self, cfg: argparse.Namespace):

        assert cfg.layer is not None
        assert cfg.phase is not None

        out_dir = lambda L, P: os.path.join(cfg.data_dir, f"L{L}", f"P{P}")
        file_path = lambda L, P: os.path.join(out_dir(L,P), f"dfs-L{L}-P{P}.csv")

        csv_path = file_path(cfg.layer, cfg.phase)

        ds = datasets.Dataset.from_csv(csv_path)

        def fix_data(data_str: str):
            data_str = data_str.replace("\n"," ").strip("[]")
            try:
                np_array = np.fromstring(data_str,sep=' ')
            except ValueError as e:
                print(f"Error parsing: {e}")
            return {"data":np_array}

        ds = ds.map(fix_data, input_columns="data")

        config_name = f"layer-{cfg.layer:02}-phase-{cfg.phase}"
        print(f"Pushing {config_name} to hub")
        ds.push_to_hub(cfg.ds_repo, config_name=config_name, split=self.split_name)


class ActivationDatasetGenerator(AbstractCommand):
    """Exports activations in CSV format for all layers and phases."""

    cfg: argparse.Namespace

    MOVE_PHASES = [
        WHITE_TO,  # 2 phase = 0
        BLACK_FROM,  # 3 phase = 1
        BLACK_TO,  # 4 phase = 2
        WHITE_FROM,  # 1 phase = 3
        PROMOTION,  # 5 phase = 4
        SPECIAL,  # 6: note addressed here
    ] = range(6)

    N_PHASES = 5 # skip SPECIAL
    START_POS = -5 # only capture state of final 5 tokens

    N_LAYERS: int = None

    def add_arguments(self, parser):
        # fmt: off
        parser.add_argument("data_dir", type=str, help="Directory where processed files are saved.") 
        parser.add_argument("ds_config", type=str, help="Hf 🤗 dataset config name (e.g., '202301')") 
        parser.add_argument("ds_repo", type=str, help="Hf 🤗 dataset repository name (e.g., 'user/repo')") 
        parser.add_argument("ds_split", type=str, help="Hf 🤗 dataset split name (e.g. 'train')") 
        parser.add_argument("model_checkpoint", type=str, help="local or Hf 🤗 model used to generate hidden state vectors")
        parser.add_argument("--start_pos", type=int, default=-5, help="Number of steps from the end of the token sequence to process.")
        # fmt: on
        return parser

    def execute(self, cfg: argparse.Namespace):

        self.cfg = cfg

        ########################
        ## Load model & tokenizer
        ########################

        model = (
            GPT2LMHeadModel.from_pretrained(cfg.model_checkpoint)
            .train(False)
            .to(torch.device("cuda"))
        )

        self.N_LAYERS = len(model.transformer.h) + 1
        tokenizer = UciTileTokenizer()

        ########################
        ## Load dataset and tokenize
        ########################

        dataset = (
            datasets.load_dataset(cfg.ds_repo, name=cfg.ds_config, split=cfg.ds_split)
            .map(self.tokenize, batched=True, fn_kwargs={"tokenizer":tokenizer})
            .map(lambda input_ids: {"num_tokens": len(input_ids)}, input_columns="input_ids")
            .sort("num_tokens")
            .filter(lambda num_tokens: num_tokens < 512, input_columns="num_tokens")
        )

        ########################
        ## Prepare paths and BufferedWriters
        ########################
        out_dir = lambda L, P: os.path.join(cfg.data_dir, f"L{L}", f"P{P}")
        file_path = lambda L, P: os.path.join(out_dir(L,P), f"dfs-L{L}-P{P}.csv")

        for L in range(self.N_LAYERS):
            for P in range(self.N_PHASES):
                os.makedirs(out_dir(L,P), exist_ok=True)

        writers: BufferedWriter = [
            [open(file_path(L, P), "a") 
                for L in range(self.N_LAYERS)] 
                for P in range(self.N_PHASES)
        ]

        print_headers = True # only once at the start

        for sample in tqdm(dataset):

            ########################
            ## Process Board state
            ########################
            transcript = sample["transcript"].lower()
            encoding = tokenizer(transcript, return_offsets_mapping=True)
            offset_mapping = encoding["offset_mapping"]
            try:
                fen_by_pos, phase_by_pos = self.get_board_fens_by_pos(transcript, offset_mapping)
            except (chess.InvalidMoveError, chess.IllegalMoveError) as ex:
                # skip any sample w/ illegal move (typically occurs b/c repetition)
                # skip any sampel w/ invalid move (typically b/c bad format on promotion)
                continue  
            fen_by_pos, phase_by_pos = fen_by_pos[cfg.start_pos:], phase_by_pos[cfg.start_pos:] 

            ########################
            ## Process Hidden States
            ########################
            hidden_states = self.transcript_to_hidden_states(transcript, tokenizer, model)[0]
            hidden_states_trimmed, seqn_start_pos_idx = self.trim_hidden_states(hidden_states, cfg.start_pos)
            indices, records = self.hidden_states_to_records(hidden_states_trimmed, seqn_start_pos_idx)

            ########################
            ## Export/append to CSV
            ########################
            df = self.records_to_df(indices, records, fen_by_pos, phase_by_pos, sample["site"])

            for L in range(self.N_LAYERS):
                for P in range(self.N_PHASES):
                    LP_subset: pd.DataFrame = df[(df["layer"] == L) & (df["phase"] == P)]
                    LP_subset.to_csv(writers[P][L], index=False, header=print_headers)
            print_headers = False

    def tokenize(self, batch, tokenizer: PreTrainedTokenizerFast):
        return tokenizer.batch_encode_plus(
                batch["transcript"],
                return_attention_mask=False,
                return_token_type_ids=False,
            )

    def transcript_to_hidden_states(self, 
        transcript_batch: Union[str | List[str]],
        tokenizer: PreTrainedTokenizerFast,
        model: GPT2LMHeadModel,
    ) -> List[torch.Tensor]:
        """
        Converts a batch of uci transcripts into a list of hidden state tensors of
        shape [batch_size, [n_layer, n_pos, d_model]]
        """
        # tokenize inputs
        encoding = tokenizer(
            transcript_batch,
            padding=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_length=True,
        ).to(torch.device("cuda"))
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # stack hidden states
        hidden_states = outputs.hidden_states
        hidden_states = torch.stack(hidden_states).permute(1, 0, 2, 3)
        hidden_states = hidden_states.to(torch.device("cpu"))

        # recompute sequence lengths ignoring padding
        lengths = encoding["length"] - sum(encoding["special_tokens_mask"].T) + 1

        # only select token positions of hidden state that were from original sequence
        hidden_states_list = [s[: lengths[i]] for i, s in enumerate(hidden_states)]

        return hidden_states_list

    def hidden_states_to_records(self, 
        hidden_state_tensors: torch.Tensor, min_pos: int
    ) -> Tuple[tuple, torch.Tensor]:
        r"""Flattens the hidden state tensor into a list of tensors.
        Iteration is like:
            original[L,P] === records[P*9+L]

        Example::

            >>> indices, records = hidden_states_to_records(output)
            >>> k = 15
            >>> L, P = indices[k]
            >>> print(f"L: {L}, P: {P}")
            L: 6, P: 1
            >>> print(sum(abs(records[k]-records[P*9+L])))
            tensor(0.)
            >>> print(sum(abs(output[L,P]-records[P*9+L])))
            tensor(0.)
        """

        n_layer, n_pos, d_model = hidden_state_tensors.shape
        records = hidden_state_tensors.permute(1, 0, 2).reshape(-1, d_model).unbind()
        indices = [(L, P + min_pos) for P in range(n_pos) for L in range(n_layer)]
        return indices, records

    def trim_hidden_states(self,
        hs: torch.Tensor, pos_start: int = -5, pos_end: int = None
    ) -> Tuple[torch.Tensor, int]:
        n_pos = hs.shape[1]
        hs = hs[:, pos_start:]
        return hs, n_pos - 5

    def diff(self, x):
        return x[1] - x[0]

    def get_board_fens_by_pos(self, transcript, offset_mapping):

        board_stack: List[chess.Board] = uci_to_board(
            transcript,
            as_board_stack=True,
            force=False,
            verbose=False,
        )

        fens_by_pos: List[str] = [board_stack[0].fen()]  # always include 1st board
        phases_by_pos: List[int] = [self.SPECIAL]
        # we must duplicate boards for each token. Sometimes, that's every 2 tokens,
        # sometimes that's every 3 (e.g., for promotions and at start tokens).
        board_pos = 0
        current_move_phase = 0
        for pos in range(1, len(offset_mapping)):
            if self.diff(offset_mapping[pos]) < 2:
                phases_by_pos.append(self.PROMOTION)
            else:
                phases_by_pos.append(current_move_phase)
                current_move_phase = (current_move_phase + 1) % 4

            prior_offset = offset_mapping[pos - 1]
            current_offset = offset_mapping[pos]

            # check for space between previous token and current token
            if prior_offset[1] != current_offset[0]:
                board_pos += 1

            fens_by_pos.append(board_stack[board_pos].fen())

        return fens_by_pos, phases_by_pos

    def records_to_df(self,
        indices: List[Tuple], records: Tuple[torch.Tensor], fen_by_pos, phase_by_pos, site
    ):
        df = pd.DataFrame(indices, columns=["layer", "pos"])
        n_layer = max(df["layer"]) + 1
        df["phase"] = [
            phase_by_pos[i // n_layer] for i in range(len(phase_by_pos) * n_layer)
        ]
        df["site"] = [site] * len(df)
        df["fen"] = [fen_by_pos[i // n_layer] for i in range(len(fen_by_pos) * n_layer)]
        df["data"] = [r.numpy() for r in records]
        return df


if __name__ == "__main__":
    main()
