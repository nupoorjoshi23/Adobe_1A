# src/run_inference.py (FINAL - With correct config loading)
import os
import json
import fitz
import torch
from transformers import AutoTokenizer, AutoConfig # <-- Import AutoConfig
from src.model import MiniLayoutLM
import numpy as np

# --- Configuration for Local Testing ---
MODEL_DIR = 'models/electra-layoutlm-full'
QUANTIZED_MODEL_PATH = 'models/electra-layoutlm-full-quantized.pth'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
LABELS = ['TITLE', 'H1', 'H2', 'H3', 'BODY', 'OTHER']

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- CRITICAL CHANGE HERE ---
# Use AutoConfig to load the configuration correctly
config = AutoConfig.from_pretrained(MODEL_DIR)
# ----------------------------

# Now, MiniLayoutLM(config) will receive the correct object type
model = MiniLayoutLM(config)
model.load_state_dict(torch.load(QUANTIZED_MODEL_PATH))
model.eval()

def normalize_bbox(bbox, width, height):
    # ... (the rest of the script is unchanged)
    return [
        int(1000 * bbox[0] / width), int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width), int(1000 * bbox[3] / height),
    ]

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    final_title, final_outline = "Untitled", []
    for page_num, page in enumerate(doc):
        words_on_page = page.get_text("words")
        if not words_on_page: continue
        words, unnormalized_bboxes = [w[4] for w in words_on_page], [w[:4] for w in words_on_page]
        tokenized = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        word_ids = tokenized.word_ids()
        aligned_bboxes = []
        for word_idx in word_ids:
            if word_idx is None: aligned_bboxes.append([0, 0, 0, 0])
            else: aligned_bboxes.append(normalize_bbox(unnormalized_bboxes[word_idx], page.rect.width, page.rect.height))
        tokenized['bbox'] = torch.tensor([aligned_bboxes], dtype=torch.long)
        with torch.no_grad(): outputs = model(**tokenized)
        predictions = np.argmax(outputs[0].numpy(), axis=2)[0]
        current_heading_text, current_heading_label, last_word_id = "", "", -1
        for i, word_id in enumerate(word_ids):
            if word_id is None or word_id == last_word_id: continue
            last_word_id = word_id
            label = LABELS[predictions[i]]
            if label not in ['BODY', 'OTHER']:
                if label != current_heading_label and current_heading_text:
                    level, text = current_heading_label.lower(), current_heading_text.strip()
                    if level == 'title':
                        if final_title == "Untitled": final_title = text
                    else: final_outline.append({"level": level, "text": text, "page": page_num + 1})
                    current_heading_text = ""
                current_heading_text += " " + words[word_id]
                current_heading_label = label
            else:
                if current_heading_text:
                    level, text = current_heading_label.lower(), current_heading_text.strip()
                    if level == 'title':
                        if final_title == "Untitled": final_title = text
                    else: final_outline.append({"level": level, "text": text, "page": page_num + 1})
                    current_heading_text, current_heading_label = "", ""
        if current_heading_text:
            level, text = current_heading_label.lower(), current_heading_text.strip()
            if level == 'title':
                if final_title == "Untitled": final_title = text
            else: final_outline.append({"level": level, "text": text, "page": page_num + 1})
    final_outline.sort(key=lambda x: x['page'])
    unique_outline, seen = [], set()
    for item in final_outline:
        if item['text'] not in seen:
            unique_outline.append(item); seen.add(item['text'])
    return {"title": final_title, "outline": unique_outline}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(INPUT_DIR, filename)
            print(f"Processing {pdf_path}...")
            try:
                result = process_pdf(pdf_path)
                output_filename = os.path.splitext(filename)[0] + '.json'
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Successfully wrote output to {output_path}")
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()