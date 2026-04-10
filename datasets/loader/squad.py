import os
import sys
from datasets import load_from_disk, load_dataset

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import create_chat_prompt
from prompt.prompt import create_prompt

def tokenize(item, tokenizer, encoder_decoder=False):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    
    # 1. Update task to match our new safety.py and context.json
    task = "safety" 
    
    # 2. Add Qwen2.5 to the shot selection logic
    shot = 0
    if "qwen2.5" in tokenizer.name_or_path.lower():
        shot = 2 # Adjust few-shot count for Qwen as needed
    elif tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 1
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 3
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 4

    # 3. Extract the text to evaluate and the target answer from PKU-SafeRLHF
    # For alignment/SFT, we usually want the model to learn to output the safest response.
    text_to_evaluate = item['prompt']
    safe_id = item['safer_response_id']
    target_answer = item[f'response_{safe_id}'] 
    
    # NOTE: If you are strictly doing the "Safe/Unsafe" classification task from step 1 
    # instead of generative SFT, change `target_answer` to the string "Safe" or "Unsafe" 
    # based on your dataset's labels.

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            text = text_to_evaluate, # Passed into safety.py's create_request
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            text = text_to_evaluate,
        )

    # 4. Tokenization (Replaced answers_generated with target_answer)
    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    
    if not encoder_decoder:
        if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
            context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
            if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
                answer_tokens = tokenizer.encode(f" {target_answer}", add_special_tokens=False)
            else:
                # For Qwen, ensuring the answer string is cleanly tokenized
                answer_tokens = tokenizer.encode(f"{target_answer}", add_special_tokens=False)
                
                # Qwen uses <|im_end|> to terminate assistant responses. 
                # If your tokenizer doesn't append it automatically here, you might need to add its token ID manually:
                if "qwen2.5" in tokenizer.name_or_path.lower():
                    answer_tokens.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
        else:
            context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
            answer_tokens = tokenizer.encode(f" {target_answer}{tokenizer.eos_token}", add_special_tokens=False)

        prompt_tokens = context_tokens + answer_tokens
        
        # Mask out the prompt tokens with -100 so the model only calculates loss on the target answer
        labels_tokens = (len(context_tokens) * [-100,]) + answer_tokens

        combined_tokens = {
            "input_ids": prompt_tokens,
            "labels": labels_tokens
        }
        return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))
        
    else:
        # Seq2Seq handling
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        labels = tokenizer.encode(target_answer, add_special_tokens=True, return_tensors="pt")[0]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }

def get_split(dataset_config, tokenizer, split):
    print(f"\n\n{os.getcwd()}\n\n")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
    if split == "validation": 
        split = "test"  

    dataset = dataset[split]

    if dataset_config.training_size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.training_size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, dataset_config.encoder_decoder), remove_columns=list(dataset.features))
    return dataset