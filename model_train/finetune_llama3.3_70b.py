import gc
import os
import torch
import wandb
import pandas as pd
from datasets import load_dataset, Features, Value, Sequence
from trl import SFTTrainer
from huggingface_hub import login
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model


login(token="")

features = Features({
    'question': Value('string'),
    'subject': Value('string'),
    'choices': Value('string'),  
    'answer': Value('string')
})

def prepare_sample(sample):
    if isinstance(sample['choices'], str):
        try:
            choices = eval(sample['choices'])
        except:
            choices = sample['choices'].strip('[]').split(',')
            choices = [c.strip().strip("'\"") for c in choices]  
    else:
        choices = sample['choices']
    
    if isinstance(choices, list):
        choices_str = str(choices)
    else:
        choices_str = str([choices])
        
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specializing in generating plausible distractors. "
                "Your task is to generate 3 incorrect but plausible distractors for the given question. "
                "The distractors should be semantically related to the context of the question and "
                "close to the correct answer, but clearly incorrect. "
                "Provide the distractors as a single list."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {sample['question']}\n"
                f"Correct Answer: {sample['answer']}\n"
                "Please provide three plausible but incorrect distractors in the form of a single list."
            ),
        },
        {
            "role": "assistant",
            "content": choices_str
        },
    ]
    return {"messages": messages}


datasets = load_dataset('csv', 
                       data_files={
                           'train': 'mmlu_train.csv',
                           'validation': 'mmlu_val.csv'
                       },
                       features=features)

print("\nFirst example from training set:")
print(datasets['train'][0])


train_dataset = datasets['train'].map(
    prepare_sample,
    remove_columns=datasets['train'].column_names,
    load_from_cache_file=False
)
val_dataset = datasets['validation'].map(
    prepare_sample,
    remove_columns=datasets['validation'].column_names,
    load_from_cache_file=False
)

print("First processed message in train dataset:")
print(train_dataset[0]["messages"])

model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)


model.resize_token_embeddings(len(tokenizer))
assert model.config.vocab_size == len(tokenizer), f"Error: Model vocab size {model.config.vocab_size} does not match tokenizer length {len(tokenizer)}"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


training_args = TrainingArguments(
    output_dir='./llama3.3-lora-ft-distractor-generator/',
    per_device_train_batch_size=2,
    gradient_accumulation_steps=64,
    num_train_epochs=3,
    learning_rate=8e-6,
    logging_dir='./logs',
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    eval_strategy="epoch",
    remove_unused_columns=False,
    bf16=True,
)

def preprocess_function(examples):
    texts = [tokenizer.apply_chat_template(messages, tokenize=False) 
             for messages in examples["messages"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )


tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False
)
tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=False
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_seq_length=1024,
    dataset_text_field="messages"
)


trainer.train()