"""flowertune-1B-v1: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    """Construct prompts."""
    output_texts = []
    # Constructing a standard Alpaca
    # (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    for i in range(len(example["instruction"])):
        text = (
            f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n"
            f"### Response: {example['response'][i]}"
        )
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_prompt_formatting(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]

    #TODO: try which is better DataCollatorForLanguageModeling, DataCollatorForCompletionOnlyLM
    # data_collator = DataCollatorForCompletionOnlyLM(
    #     response_template_ids, tokenizser=tokenizer, padding_free=True,
    # )


    #using default data_collator DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return tokenizer, data_collator, formatting_prompts_func


def formatting(dataset):
    """Format dataset."""
    dataset["instruction"] = dataset["instruction"] + " " + dataset["input"]
    return dataset


def reformat(dataset, llm_task):
    """Reformat datasets to have `instruction`, `input` and `response` columns."""

    # Some datasets (e.g. databricks/databricks-dolly-15k) use "context" as the
    # input column.
    # *NOTE: change the "context" to "input"
    if "context" in dataset.column_names:
        dataset = dataset.rename_column("context", "input")

    # # The Code Alpaca dataset uses "output" as the response column.
    # if "output" in dataset.column_names:
    #     dataset = dataset.rename_column("output", "response")

    # if llm_task in ["finance", "code"] and "input" in dataset.column_names:
    #     dataset = dataset.map(formatting, remove_columns=["input"])

    # if llm_task == "medical" and "input" in dataset.column_names:
    #     dataset = dataset.remove_columns(["instruction"])
    #     dataset = dataset.rename_column("input", "instruction")

    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        #partitioner = IidPartitioner(num_partitions=num_partitions)

        # * NOTE: using pathological partitioner for non-iid data
        partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                              seed=42,
                                              partition_by="category",
                                              class_assignment_mode="first-deterministic",
                                              num_classes_per_partition=1,
        )
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = reformat(client_trainset, llm_task="dolly")
    return client_trainset

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
