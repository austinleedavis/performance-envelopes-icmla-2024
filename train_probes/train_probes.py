"""
Trains a probe on a specific layer and phase of data.


Usage:
    Single-run:
    ```
        python train_probes/train_probes.py --config_yaml "train_probes/configs/12x12x768/12x12x768_to_color.yaml"
    ```

    Multi-run:
    ```
    set phases 0 2 3 4
    for phase in $phases
        for layer in (seq -w 0 12)
            python train_probes/train_probes.py --config_yaml "train_probes/configs/12x12x768/12x12x768_to_color.yaml" --layer $layer --phase $phase
        end
    end
    for phase in (seq -w 2 3)
        for layer in (seq -w 0 12)
            python train_probes/train_probes.py --config_yaml "train_probes/configs/12x12x768/12x12x768_to_piece.yaml" --layer $layer --phase $phase
        end
    end
    ```
"""


# %%
import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import chess.svg
import datasets
import evaluate
import numpy as np
import plotly_express as px
import torch
from transformers import (HfArgumentParser, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)

import chess
import wandb
from modeling import board_state_functions as bsf
from modeling.linear_probe import *


def main():

    ##############################
    # Parse Args
    ##############################

    cfg, training_args, args_yml_path = parse_args()

    model_name = "-".join([cfg.state_fn_name, f"L{cfg.layer:02d}", f"P{cfg.phase}"])

    if "wandb" in training_args.report_to:
        wandb_run = wandb.init(
            name=model_name,
            resume=cfg.wandb_resume,
            id=cfg.run_id,
            config={"run_config": vars(cfg), "training_args:": vars(training_args)},
            project=cfg.wandb_project,
        )
        if cfg.do_train:
            wandb_run.log_code(
                os.path.dirname(args_yml_path),
                f"{wandb_run.name}.yaml",
                include_fn=lambda x: x.endswith(".yaml"),
            )
    ##############################
    # Setup output Directories
    ##############################
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Saving outputs to {cfg.output_dir}")

    ##############################
    # Load dataset
    ##############################

    ds = (
        datasets.load_dataset(
            path=cfg.dataset_repo_id, name=cfg.dataset_config, split="train"
        )
        .with_format("torch")
        .map(
            lambda x: {MODEL_OUT: cfg.state_fn.map(x, "fen")},
            desc="Apply state mapping",
        )
        .rename_column("data", MODEL_IN)
        # .select_columns([MODEL_IN,MODEL_OUT])
        .train_test_split(train_size=0.95, shuffle=True, seed=42)
    )

    def custom_collate_fn(data: list[dict[str, torch.Tensor]]):
        return {
            MODEL_IN: torch.stack([f[MODEL_IN] for f in data], dim=0),
            MODEL_OUT: torch.stack([f[MODEL_OUT] for f in data], dim=0).type(
                torch.float32
            ),
        }

    ##############################
    # Initialize the probe model
    ##############################
    if cfg.pretrain_checkpoint:
        model = MulticlassProbeModel.from_pretrained(cfg.pretrain_checkpoint)
        num_classes = model.config.out_features
        num_submodules = model.config.num_submodules
    else:
        hidden_shape: int = ds["train"][0][MODEL_IN].shape[0]  # 768
        output_shape = ds["train"][0][MODEL_OUT].shape
        if len(output_shape) == 2:
            num_classes, num_submodules = output_shape
        else:
            # Reserved for when doing predictions on game-level data
            NotImplementedError("No reason for this yet.")

        model_config = MulticlassProbeConfig(
            in_features=hidden_shape,
            out_features=num_classes,
            num_submodules=num_submodules,
        )
        model = MulticlassProbeModel(model_config)

    print(model)

    ##############################
    # Training
    ##############################
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=custom_collate_fn,
        callbacks=[
            PlotsCallback(
                dataset=ds,
                class_labels=cfg.state_fn.get_class_labels(),
                wandb_run=wandb_run if "wandb" in training_args.report_to else None,
            ),
            # EarlyStoppingCallback(early_stopping_patience=5),
        ],
        compute_metrics=compute_metrics,
    )

    # Train/Eval the model
    if cfg.do_train:
        trainer.train()
        
        model.name_or_path = model_name

        model.save_pretrained(cfg.output_dir)

        wandb_run.log_model(path=cfg.output_dir, name=model_name)
        
    else:
        wandb_run.log(trainer.evaluate())

    wandb.finish()


############################
############################
##
## Helper Methods Sections
##
############################
############################


############################
# Metrics
############################

EXPERIMENT_UUID: str = (
    np.random.randint(low=65, high=90, size=15, dtype="int32").view(f"U{15}").item()
)
"""Required for parallel or distributed use of evaluate APIs"""

EVAL_ACC = evaluate.load("accuracy", experiment_id=EXPERIMENT_UUID)
EVAL_ROC = evaluate.load("roc_auc", experiment_id=EXPERIMENT_UUID)
EVAL_PRECISION = evaluate.load(
    "precision",
    experiment_id=EXPERIMENT_UUID,
)
EVAL_RECALL = evaluate.load("recall", experiment_id=EXPERIMENT_UUID)
EVAL_F1 = evaluate.load("f1", experiment_id=EXPERIMENT_UUID)


def compute_metrics(eval_pred):
    """Inputs are a tuple of tensors (logs,labels) each with
    shape: [num_samples, num_classes, num_submodules]. When
    we want to do predictions, we do argmax over the class dimension
    to choose the class index with the highest logit value. This
    results in a shape [num_samples, 1, num_submodules], which is
    flattened into a 1D tensor shaped: [num_samples*num_submodules].

    ## Averaging:

    Note: A thorough description of micro/maco/weighted is available at: [here](http://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example)
    Precision and recall use **weighted** averages because pawns and blanks far outweigh other classes.

    """
    logits: torch.Tensor
    labels: torch.Tensor
    logits, labels = eval_pred

    predictions = logits.argmax(1).flatten()  # [num_samples*num_submodules]
    references = labels.argmax(1).flatten()  # [num_samples*num_submodules]

    # wandb.log({"austin_roc": wandb_roc_curve(labels, logits)})

    return {
        **EVAL_ACC.compute(predictions=predictions, references=references),
        **EVAL_ROC.compute(
            prediction_scores=logits.flatten(), references=labels.flatten()
        ),
        **EVAL_PRECISION.compute(
            predictions=predictions, references=references, average="weighted"
        ),
        **EVAL_RECALL.compute(
            predictions=predictions, references=references, average="weighted"
        ),
        **EVAL_F1.compute(
            predictions=predictions, references=references, average="weighted"
        ),
    }


############################
# State functions
############################

STATE_FUNCTIONS = {
    "to_color": bsf.ToColor(),
    "to_piece": bsf.ToPiece(),
    "to_piece_by_color": bsf.ToPieceByColor(),
}


############################
# Config and Argparse
############################

@dataclass(kw_only=True)
class Config:
    """Config used to define a specific training run"""

    layer: int
    phase: int
    state_fn_name: str
    original_output_dir: str = ""
    output_dir: str = ""
    wandb_project: str = "probe_training"
    chess_model: str = "austindavis/chess-gpt2-uci-8x8x512"
    dataset_repo_id: str = "austindavis/chess-gpt2-hiddenstates-512"
    run_id: str = None  # Only use if resuming run
    dataset_config: str = None  # computed at runtime
    state_fn: bsf.StateFunctionBase = None  # computed at runtime
    do_train: bool = True
    pretrain_checkpoint: str = None
    wandb_resume: str = "never"
    """Options: `"allow"`, `"must"`, `"never"`, `"auto"` or `None`. Defaults to `None`."""

    def set_layer(self, layer: int):
        self.layer = layer
        self._update_fields_()

    def set_phase(self, phase: int):
        self.phase = phase
        self._update_fields_()

    def __post_init__(self):
        self.original_output_dir = self.output_dir
        self._update_fields_()
        self.state_fn = STATE_FUNCTIONS[self.state_fn_name]
    
    def _update_fields_(self):
        self.output_dir = os.path.join(
            self.original_output_dir,
            str(self.state_fn_name),
            f"layer-{self.layer:02d}",
            f"phase-{self.phase}",
        )
        self.dataset_config = f"layer-{self.layer:02}-phase-{self.phase}"

def parse_args() -> Tuple[Config, TrainingArguments]:
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_yaml", required=True)
    cli_parser.add_argument("--layer", required=False, type=int, help="If provided, this will override the config layer value.")
    cli_parser.add_argument("--phase", required=False, type=int, help="If provided, will override the config's phase value.")
    if is_notebook():
        config_yaml = input("Provide the path to the config yaml. (Default: 'train_probes/configs/interactive/')")
        if len(config_yaml) == 0:
            config_yaml = "8x8x512/8x8x512_interactive.yaml"
        args = cli_parser.parse_args(["--config_yaml", config_yaml])
    else:
        args = cli_parser.parse_args()

    training_config: Config = HfArgumentParser([Config]).parse_yaml_file(
        args.config_yaml, allow_extra_keys=True
    )[0]

    if args.layer is not None:
        training_config.set_layer(args.layer)
    
    if args.phase is not None:
        training_config.set_phase(args.phase)
    

    training_args: TrainingArguments = HfArgumentParser(
        [TrainingArguments]
    ).parse_yaml_file(args.config_yaml, True)[0]

    training_args.output_dir = training_config.output_dir

    return training_config, training_args, args.config_yaml

def is_notebook() -> bool:
    try:
        shell = globals()["get_ipython"].__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        elif shell == "function":
            return True
        else:
            return False  # Other type (?)
    except (NameError, KeyError):
        return False  # Probably standard Python interpreter


class PlotsCallback(TrainerCallback):
    """
    Callback to generate and log plots during the training process.

    Attributes:
        data_source_labels (list): 
        fig_size (int): Size of the figure for plotting.
        need_board_svg (bool): Flag to determine if the chess board SVG needs to be logged.
        ds (datasets.Dataset): Dataset used for evaluation.
        num_classes (int): Number of classes for the classification task.
        classes (list): List of class labels based on the number of classes.
        wandb_run (wandb.run): Weights and Biases run object for logging.
    """

    _data_source_labels: list[str]
    """Labels for figure to indicate source of the subplot (either prediction, difference, or ground truth)"""
    _class_labels: list[str]
    """List of class labels based on the number of classes."""
    _fig_size:int = 600
    """Size of the figure for plotting"""
    _need_board_svg = True
    """Flag to determine if the chess board SVG needs to be logged."""

    def __init__(
        self, *, dataset: datasets.Dataset, class_labels: int, wandb_run=None
    ):
        """
        Initializes the PlotsCallback with the dataset, number of classes, and optional Weights and Biases run.
        
        Args:
            dataset (datasets.Dataset): The dataset used for evaluation.
            num_classes (int): The number of classes for the classification task.
            wandb_run (wandb.run, optional): The Weights and Biases run for logging.
        
        Raises:
            ValueError: If the number of classes is not valid.
        """
        self._class_labels = class_labels
        self.ds = dataset
        self.num_classes = len(class_labels)
        self._data_source_labels = ["y", "𝚫", "ŷ"]

        if class_labels is None:
            raise ValueError("Not valid number of pieces")

        self.wandb_run = wandb_run

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        """
        Called during the evaluation phase to generate and log plots.
        
        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): The current state of the Trainer.
            control (TrainerControl): The control object for managing the training process.
            metrics (dict): The evaluation metrics.
            model (torch.nn.Module): The model being trained.
        """
        self.make_plots(
            model,
            ds=self.ds,
            k=0,
            num_classes=self.num_classes,
            state=state,
            metrics=metrics,
            fig_size=self._fig_size
        )
        return super().on_evaluate(args, state, control, **kwargs)

    def make_plots(self, model, ds, k, num_classes, state=None, metrics=None, fig_size=300):
        """
        Generates and logs plots to Weights and Biases using the model predictions.
        
        Args:
            model (torch.nn.Module): The model being trained.
            ds (datasets.Dataset): The dataset used for evaluation.
            k (int): Index of the data point to evaluate.
            num_classes (int): The number of classes for the classification task.
            state (TrainerState, optional): The current state of the Trainer.
            metrics (dict, optional): The evaluation metrics.
            fig_size (int, optional): The size of the figure for plotting.
        """
        output = (
            model(ds["test"][k]["input"].to(torch.device("cuda")))
            .detach()
            .to("cpu")
            .softmax(-1)
        )
        truth = ds["test"][k]["labels"].to("cpu").permute(1, 0)
        diff = output - truth
        combined = torch.stack([output, diff, truth], -1)
        fig = px.imshow(
                combined.permute(0,2,1).reshape(8, 8, 3 * num_classes).flip(0),
                x=list("abcdefgh"),
                y=list("87654321"),
                facet_col=-1,
                facet_col_wrap=num_classes,
                color_continuous_midpoint=0,
                color_continuous_scale="RdBu",
                zmin=-1.0,
                zmax=1.0,
                facet_col_spacing=0,
                facet_row_spacing=0,
                width=fig_size,
                height=fig_size,
                aspect="equal",
                title=f"Model Performance (Step: {state.global_step}, eval_loss: {metrics['eval_loss']:0.4f})"
            )
        
        """Facet labels are uninformative. They need to be updated individually. We use the counter below
        to determine which facet we're updating and select the appropriate labels."""
        self.facet_counter = 0
        fig.for_each_annotation(self.update_facet_headers)

        board = chess.svg.board(chess.Board(ds['test'][k]['fen']),size=fig_size/2)

        if self.wandb_run:
            if self._need_board_svg:
                # Only log chess board once, at the start of training.
                table = wandb.Table(columns=["Board"])
                table.add_data(wandb.Html(board))
                self.wandb_run.log({"Board":table})
                self._need_board_svg = False

            self.wandb_run.log({'ProbePredictions':fig})

    def update_facet_headers(self, a):
        src_lbl = self._data_source_labels[self.facet_counter // self.num_classes]
        cls_lbl = self._class_labels[self.facet_counter % self.num_classes]
        a.update(text=f"{src_lbl} {cls_lbl}")
        self.facet_counter += 1


# %%

if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        wandb.finish(0)

# %%
