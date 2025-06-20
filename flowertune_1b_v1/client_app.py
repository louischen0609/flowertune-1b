"""flowertune-1B-v1: A Flower / FlowerTune app."""

import os
import warnings
import atexit
from typing import Dict, Tuple

import wandb

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig, OmegaConf

from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_1b_v1.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_1b_v1.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
os.environ["WANDB_DISABLE_SYSTEM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Flower client for LLM fine-tuning."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        wandb_cfg,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = TrainingArguments(**train_cfg.training_arguments)
        if wandb_cfg.get("use_wandb", False):
            self.training_arguments.report_to = "wandb"
        else:
            self.training_arguments.report_to = "none"
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.wandb_cfg = wandb_cfg

        # instantiate model
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config["save_path"]

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Initialize wandb run for this client
    if cfg.get("wandb", {}).get("use_wandb", False):
        run_name_prefix = cfg.wandb.get("run_name_prefix", "client")
        run_name = f"{run_name_prefix}-{partition_id}"
        wandb.init(
            project=cfg.wandb.get("project", "flower-client"),
            name=run_name,
            entity=cfg.wandb.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(x_disable_stats=True),
        )
        atexit.register(wandb.finish)

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        cfg.get("wandb", {}),
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
