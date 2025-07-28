# # src/train.py (UPDATED for ELECTRA base model)
# import json
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, Trainer, TrainingArguments
# from sklearn.model_selection import train_test_split
# from src.model import MiniLayoutLM

# # --- Configuration ---
# DATA_FILE = 'data/processed/unified_data.jsonl'
# # --- CRITICAL CHANGE: Swapping the base model to ELECTRA ---
# BASE_MODEL = 'google/electra-small-discriminator'
# MODEL_OUTPUT_DIR = 'models/electra-layoutlm' # New directory for the new model
# LABELS = ['TITLE', 'H1', 'H2', 'H3', 'BODY', 'OTHER']
# label2id = {label: i for i, label in enumerate(LABELS)}
# id2label = {i: label for i, label in enumerate(LABELS)}

# # --- Load and Preprocess Data ---
# # AutoTokenizer will work reliably with ELECTRA's fast tokenizer
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# dataset = load_dataset('json', data_files=DATA_FILE, split='train')
# dataset = dataset.select(range(10000))

# def preprocess_data(examples):
#     words = examples['words']
#     labels = examples['labels']
#     bboxes = examples['bboxes']

#     tokenized_inputs = tokenizer(words, is_split_into_words=True, padding='max_length', truncation=True, max_length=512)
    
#     aligned_labels = []
#     aligned_bboxes = []
#     for i, label_list in enumerate(labels):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         previous_word_idx = None
#         label_ids = []
#         bbox_ids = []
#         for word_idx in word_ids:
#             if word_idx is None:
#                 label_ids.append(-100)
#                 bbox_ids.append([0, 0, 0, 0])
#             elif word_idx != previous_word_idx:
#                 label_ids.append(label2id[label_list[word_idx]])
#                 bbox_ids.append(bboxes[i][word_idx])
#             else:
#                 label_ids.append(-100)
#                 bbox_ids.append(bboxes[i][word_idx])
#             previous_word_idx = word_idx
#         aligned_labels.append(label_ids)
#         aligned_bboxes.append(bbox_ids)

#     tokenized_inputs["labels"] = aligned_labels
#     tokenized_inputs["bbox"] = aligned_bboxes
#     return tokenized_inputs

# processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
# train_dataset, eval_dataset = processed_dataset.train_test_split(test_size=0.1).values()

# # --- Model and Trainer ---
# model = MiniLayoutLM.from_pretrained(
#     BASE_MODEL, 
#     num_labels=len(LABELS),
#     id2label=id2label,
#     label2id=label2id
# )

# training_args = TrainingArguments(
#     output_dir='models/electra-layoutlm-training-output', # Temporary dir for checkpoints
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     fp16=False,
#     learning_rate=3e-5,
#     logging_dir='./logs',
#     logging_steps=100,
#     eval_strategy="epoch", # Corrected argument name for modern transformers
#     save_strategy="epoch",
#     load_best_model_at_end=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
# )

# print(f"🚀 Starting model training with {BASE_MODEL} base...")
# trainer.train()

# print("✅ Training complete. Saving final model...")
# trainer.save_model(MODEL_OUTPUT_DIR)
# print(f"Model saved to {MODEL_OUTPUT_DIR}")


# src/train.py (FINAL - This is the correct version)
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from src.model import MiniLayoutLM

# --- Configuration ---
DATA_FILE = 'data/processed/unified_data.jsonl'
BASE_MODEL = 'models/electra-base-local' # Pointing to our safe, local base model
MODEL_OUTPUT_DIR = 'models/electra-layoutlm-full'
LABELS = ['TITLE', 'H1', 'H2', 'H3', 'BODY', 'OTHER']
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# --- Load and Preprocess Data ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
dataset = load_dataset('json', data_files=DATA_FILE, split='train')

# Training on the FULL dataset

def preprocess_data(examples):
    words, labels, bboxes = examples['words'], examples['labels'], examples['bboxes']
    tokenized_inputs = tokenizer(words, is_split_into_words=True, padding='max_length', truncation=True, max_length=512)
    aligned_labels, aligned_bboxes = [], []
    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx, label_ids, bbox_ids = None, [], []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100); bbox_ids.append([0, 0, 0, 0])
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_list[word_idx]]); bbox_ids.append(bboxes[i][word_idx])
            else:
                label_ids.append(-100); bbox_ids.append(bboxes[i][word_idx])
            previous_word_idx = word_idx
        aligned_labels.append(label_ids); aligned_bboxes.append(bbox_ids)
    tokenized_inputs["labels"], tokenized_inputs["bbox"] = aligned_labels, aligned_bboxes
    return tokenized_inputs

processed_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
train_dataset, eval_dataset = processed_dataset.train_test_split(test_size=0.1).values()

# --- Model and Trainer ---
model = MiniLayoutLM.from_pretrained(
    BASE_MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir='models/electra-layoutlm-training-output',
    num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=16,
    fp16=True, learning_rate=3e-5, logging_dir='./logs', logging_steps=100,
    evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=eval_dataset, tokenizer=tokenizer,
)

print(f"🚀 Starting GPU model training with local {BASE_MODEL} on the FULL dataset...")
trainer.train()

print("✅ Training complete. Saving final model...")
trainer.save_model(MODEL_OUTPUT_DIR)
print(f"Model saved to {MODEL_OUTPUT_DIR}")