import argparse
import math
import os
import random
from dataclasses import dataclass

import pandas as pd
import wandb
from datasets import ClassLabel, load_dataset, load_from_disk
from huggingface_hub import notebook_login
from IPython.display import HTML, display
from transformers import (AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel,
                          HfArgumentParser, PreTrainedTokenizerFast, Trainer,
                          TrainingArguments)
from transformers.generation import GenerationConfig

from modeling.uci_tokenizers import UciTileTokenizer


@dataclass
class TrainingConfig:
    output_dir: str = None
    dataset_dir: str = None
    config_dir: str = None
    hf_user: str = "austindavis"
    wandb_project = "huggingface"
    block_size: int = 512
    do_finetune: bool = False
    pretrain_checkpoint: str = None
    resume_training: bool = False
    resume_id: str = None
    model_checkpoint: str = None  # Computed post-init
    resume_from_checkpoint = None  # Computed post-init

    def __post_init__(self):
        self.model_checkpoint = f"{self.hf_user}/{self.output_dir}"
        self.resume_from_checkpoint = True if self.resume_training else None
        assert (
            not self.do_finetune or self.pretrain_checkpoint is not None
        ), "You must specify `pretrain_checkpoint` when `do_finetune` is True"

        assert (
            not self.resume_training or self.resume_id is not None
        ), "You must specify `resume_id` when `resume_training` is True"


def main():
    ################################################
    ## Parse Args
    ################################################
    args_path = parse_args_path()

    training_config: TrainingConfig = HfArgumentParser(
        [TrainingConfig]
    ).parse_json_file(args_path, True)[0]
    training_args: TrainingArguments = HfArgumentParser(
        [TrainingArguments]
    ).parse_json_file(args_path, True)[0]

    print(training_args)
    print(training_config)

    ################################################
    ## Prepare wandb and hf reporting
    ################################################
    notebook_login(new_session=False)
    if is_notebook():
        os.environ["WANDB_NOTEBOOK_NAME"] = globals()["__vsc_ipynb_file__"]

    if training_config.resume_training:
        assert training_config.resume_id is not None
        wandb.init(
            id=training_config.resume_id,
            resume="must",
            project=training_config.wandb_project,
        )
    else:
        wandb.init(project=training_config.wandb_project)

    ################################################
    ## Prepare dataset
    ################################################

    tokenizer = UciTileTokenizer()
    tokenizer.push_to_hub(
        repo_id=training_config.model_checkpoint, commit_message="Upload Tokenizer"
    )

    try:
        lm_datasets = load_from_disk(training_config.dataset_dir)
    except FileNotFoundError as e:
        print(
            f"WARNING! The pre-processed dataset was not found at {training_config.dataset_dir}"
        )
        response = input("Would you like to pre-process the data? [y/N]?")
        confirmed = False

        if response.lower().startswith("y"):
            response = input(
                "Are you absolutely certain? This operation will remove the on-disk dataset [y/N]?"
            )
            if response.lower().startswith("y"):
                confirmed = True

        if confirmed:
            lm_datasets = rebuild_dataset(
                training_config.dataset_dir,
                tokenizer=tokenizer,
                block_size=training_config.block_size,
            )
        else:
            print("Negative response receive. Terminating training.")
            wandb.run.finish(exit_code=-1, quiet=True)
            exit(-1)

    ################################################
    ## Init Model
    ################################################

    model = None

    if training_config.do_finetune:
        model = AutoModelForCausalLM.from_pretrained(
            training_config.pretrain_checkpoint
        )
    else:
        gpt_config_path = os.path.join(training_config.config_dir, "config.json")
        gpt_config = GPT2Config.from_json_file(gpt_config_path)
        model = GPT2LMHeadModel(gpt_config)
    model

    GenerationConfig(
        max_new_tokens=model.config.n_ctx,
        max_length=model.config.n_ctx,
        do_sample=True,
        temperature=0.0001,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ).push_to_hub(
        training_config.model_checkpoint, commit_message="Upload Generation Config"
    )

    ################################################
    ## Train
    ################################################

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
    )

    trainer.train(
        resume_from_checkpoint=True if training_config.resume_training else None
    )

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.push_to_hub()

    wandb.run.finish()


def rebuild_dataset(
    output_folder: str, tokenizer: PreTrainedTokenizerFast, block_size: int
):

    datasets = load_dataset("austindavis/lichess_uci", name="202306")
    datasets = datasets.rename_column(
        original_column_name="transcript", new_column_name="text"
    ).remove_columns("site")

    def show_random_elements(dataset, num_examples=10):
        assert num_examples <= len(
            dataset
        ), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset) - 1)
            while pick in picks:
                pick = random.randint(0, len(dataset) - 1)
            picks.append(pick)

        df = pd.DataFrame(dataset[picks])
        for column, typ in dataset.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        display(HTML(df.to_html()))

    show_random_elements(datasets["train"])

    tokenized_datasets = datasets.map(
        tokenizer, batched=True, remove_columns=["text"], input_columns="text"
    )

    print(datasets)
    print(tokenized_datasets["train"][1]["input_ids"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=16,
        load_from_cache_file=False,
    )
    lm_datasets.save_to_disk(output_folder)

    print(lm_datasets)
    print("Concatenated Samples:")
    print(tokenizer.decode(lm_datasets["train"][0]["input_ids"]))
    print("Tokens Concated: ", lm_datasets["train"][:2]["input_ids"])
    print("Tokens Original: ", tokenizer(datasets["train"][0:5]["text"])["input_ids"])
    print(f'Original Sample 0: {datasets["train"][0]["text"]}')

    return lm_datasets


def parse_args_path():
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_dir", required=True)
    if is_notebook():
        args = cli_parser.parse_args(["--config_dir", input("Provide a config_path")])
    else:
        args = cli_parser.parse_args()

    args_json = os.path.join(args.config_dir, "args.json")
    return args_json


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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt... Terminating wandb run.")
        wandb.run.finish(-1)
        exit()
