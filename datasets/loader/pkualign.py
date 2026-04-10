import os
import sys

from datasets import load_dataset, load_from_disk

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import create_chat_prompt, create_prompt


def tokenize(item, tokenizer, encoder_decoder=False):
    task = "safety"
    safe_id = item["safer_response_id"]
    target_answer = item[f"response_{safe_id}"]
    text_to_evaluate = item["prompt"]

    user_prompt = create_chat_prompt(
        task, few_shot=2, text=text_to_evaluate, chat_template=None
    )

    conversation = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": target_answer},
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False
    )

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def get_split(dataset_config, tokenizer, split):
    print(f"\n\n{os.getcwd()}\n\n")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
    if split == "validation":
        split = "test"

    dataset = dataset[split]

    if dataset_config.training_size < 1:
        dataset = dataset.select(
            range(int(len(dataset) * dataset_config.training_size))
        )
    dataset = dataset.map(
        lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder),
        batched=True,
        remove_columns=list(dataset.features),
    )
    return dataset
