import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import json
import logging
import numpy as np
from tqdm import tqdm
import csv
import ast  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.3-70B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()


# Load MMLU test
mmlu_dataset = load_dataset("cais/mmlu", "all")["test"]
mmlu_df = mmlu_dataset.to_pandas()
logging.info(f"MMLU 'test' dataset loaded. Total examples: {len(mmlu_df)}")

# Load DGEN dataset
dgen_file_path = "mcg_llama3.3_70B_inference_filtered_updated.csv"
mmlu_dgen_df = pd.read_csv(dgen_file_path)
logging.info(f"MMLU-DGEN dataset loaded. Total examples: {len(mmlu_dgen_df)}")

merged_df = pd.merge(mmlu_df, mmlu_dgen_df, on=["subject", "question"], suffixes=("_mmlu", "_dgen"))
logging.info(f"Merged dataset total examples: {len(merged_df)}")

# Answer of choices mapping
answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def parse_choices(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(x)
            except Exception as e:
                logging.warning(f"Failed to parse choices: {x} | Error: {e}")
                return []
    elif isinstance(x, (list, np.ndarray)):
        return x.tolist() if isinstance(x, np.ndarray) else x
    else:
        logging.warning(f"Unexpected type for choices: {type(x)} | Value: {x}")
        return []

# Logit computation functions
def get_logits_batch(questions, choices_list, model, tokenizer, device):
    input_texts = []
    for question, choices in zip(questions, choices_list):
        if len(choices) != 4:  ##In the paper, we confirmed that there are 4 choices for all questions.
            logging.warning(f"Expected 4 choices, got {len(choices)} for question: {question}"). 
            choices = choices[:4] + [""]*(4 - len(choices)) if len(choices) < 4 else choices[:4]
        input_texts.append(
            "Choose one: A, B, C, or D for the following question.\n"
            "Do not provide additional explanation.\n\n"
            f"Q: {question}\n"
            "Options:\n"
            f"A: {choices[0]}\n"
            f"B: {choices[1]}\n"
            f"C: {choices[2]}\n"
            f"D: {choices[3]}"
        )
    
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]  # Last token logits
    
    # Extract logits for A, B, C, D
    result_logits_batch = []
    for i, logits_row in enumerate(logits):
        result_logits = []
        for choice in ["A", "B", "C", "D"]:
            token_ids = tokenizer.encode(choice, add_special_tokens=False)
            if not token_ids:
                logging.warning(f"Tokenization failed for choice '{choice}' in question {i}")
                result_logits.append(float('-inf'))  # Assign -inf if tokenization fails
                continue
            token_id = token_ids[0]
            result_logits.append(logits_row[token_id].item())
        result_logits_batch.append(result_logits)
    return result_logits_batch


output_csv_path = "logit_llama70b.csv"
fieldnames = ["subject", "question", "mmlu_correct_logit", "mmlu_distractor_logits", "dgen_correct_logit", "dgen_distractor_logits"]

batch_size = 2  


with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    logging.info("Starting logit computation...")
    for batch_start in tqdm(range(0, len(merged_df), batch_size), desc="Processing Batches"):
        batch_end = batch_start + batch_size
        batch_df = merged_df.iloc[batch_start:batch_end]

        try:
            subjects = batch_df["subject"].tolist()
            questions = batch_df["question"].tolist()
            
            # Parse choices 
            choices_mmlu = batch_df["choices_mmlu"].apply(parse_choices).tolist()
            choices_dgen = batch_df["choices_dgen"].apply(parse_choices).tolist()
            
            answers_mmlu = batch_df["answer_mmlu"].tolist()
            answers_dgen = batch_df["answer_dgen"].tolist()

            # Compute logits for MMLU and DGEN
            logits_mmlu_batch = get_logits_batch(questions, choices_mmlu, model, tokenizer, device)
            logits_dgen_batch = get_logits_batch(questions, choices_dgen, model, tokenizer, device)

            for i in range(len(batch_df)):
                if len(logits_mmlu_batch[i]) != 4:
                    logging.warning(f"MMLU logits length mismatch for subject '{subjects[i]}'. Expected 4, got {len(logits_mmlu_batch[i])}. Skipping.")
                    continue
                if len(logits_dgen_batch[i]) != 4:
                    logging.warning(f"DGEN logits length mismatch for subject '{subjects[i]}'. Expected 4, got {len(logits_dgen_batch[i])}. Skipping.")
                    continue

                # Map answers to indices
                mmlu_answer = answers_mmlu[i]
                dgen_answer = answers_dgen[i]

                # Convert MMLU answer to index
                if isinstance(mmlu_answer, str):
                    mmlu_answer = mmlu_answer.upper()
                    mmlu_answer_idx = answer_map.get(mmlu_answer, None)
                elif isinstance(mmlu_answer, int):
                    mmlu_answer_idx = mmlu_answer if mmlu_answer in answer_map.values() else None
                else:
                    mmlu_answer_idx = None

                if mmlu_answer_idx is None:
                    logging.warning(f"Invalid MMLU answer '{mmlu_answer}' for subject '{subjects[i]}'. Skipping.")
                    continue

                # Convert DGEN answer to index
                if isinstance(dgen_answer, str):
                    dgen_answer = dgen_answer.upper()
                    dgen_answer_idx = answer_map.get(dgen_answer, None)
                elif isinstance(dgen_answer, int):
                    dgen_answer_idx = dgen_answer if dgen_answer in answer_map.values() else None
                else:
                    dgen_answer_idx = None

                if dgen_answer_idx is None:
                    logging.warning(f"Invalid DGEN answer '{dgen_answer}' for subject '{subjects[i]}'. Skipping.")
                    continue

                # Extract correct logits and distractors
                mmlu_correct_logit = logits_mmlu_batch[i][mmlu_answer_idx]
                mmlu_distractor_logits = [
                    logit for idx, logit in enumerate(logits_mmlu_batch[i]) if idx != mmlu_answer_idx
                ]

                dgen_correct_logit = logits_dgen_batch[i][dgen_answer_idx]
                dgen_distractor_logits = [
                    logit for idx, logit in enumerate(logits_dgen_batch[i]) if idx != dgen_answer_idx
                ]

                # Ensure distractor logits are lists
                if isinstance(mmlu_distractor_logits, np.ndarray):
                    mmlu_distractor_logits = mmlu_distractor_logits.tolist()
                if isinstance(dgen_distractor_logits, np.ndarray):
                    dgen_distractor_logits = dgen_distractor_logits.tolist()

                # Sort distractor logits in descending order 
                mmlu_distractor_logits.sort(reverse=True)
                dgen_distractor_logits.sort(reverse=True)

                writer.writerow({
                    "subject": subjects[i],
                    "question": questions[i],
                    "mmlu_correct_logit": mmlu_correct_logit,
                    "mmlu_distractor_logits": json.dumps(mmlu_distractor_logits),
                    "dgen_correct_logit": dgen_correct_logit,
                    "dgen_distractor_logits": json.dumps(dgen_distractor_logits),
                })

        except Exception as e:
            logging.error(f"Error processing batch {batch_start}-{batch_end}: {e}")

logging.info(f"Logit computation completed. Results saved to {output_csv_path}")
