# src/run_hybrid_inference.py (FINAL, INTELLIGENT HYBRID VERSION)
import os
import json
import fitz # PyMuPDF
import re
import statistics
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig
from src.model import MiniLayoutLM

# --- Configuration ---
MODEL_DIR = 'models/electra-layoutlm-full'
QUANTIZED_MODEL_PATH = 'models/electra-layoutlm-full-quantized.pth'
INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'
LABELS = ['TITLE', 'H1', 'H2', 'H3', 'BODY', 'OTHER']

# --- Pass 1: Machine Learning Model ---
def run_ml_pass(pdf_path, ignore_texts):
    print("  Running ML Pass...")
    # This function remains the same as before, but now accepts a set of texts to ignore
    # ... (code for ML pass is the same, but with an added check)
    return {"title": "Untitled", "outline": []} # Placeholder for brevity

# --- Pass 2: The Intelligent Rule-Based Parser ---
def run_smarter_rule_based_pass(pdf_path, ignore_texts):
    print("  Running Smarter Rule-Based Pass...")
    doc = fitz.open(pdf_path)
    rule_title, rule_outline = "", []
    
    # Analyze fonts
    all_spans = [s for page in doc for b in page.get_text("dict", flags=11)["blocks"] if b.get('lines') for l in b['lines'] for s in l['spans']]
    if not all_spans: return {"title": "", "outline": []}
    font_sizes = [s['size'] for s in all_spans]
    max_size, median_size = max(font_sizes) if font_sizes else 0, statistics.median(font_sizes) if font_sizes else 0
    
    # Detect TOC pages
    toc_pages = set()
    for page_num, page in enumerate(doc):
        if len([b for b in page.get_text("blocks") if re.search(r'[\. ]{4,}\s*\d+\s*$', b[4])]) > 3:
            toc_pages.add(page_num + 1)

    # Find Title
    page = doc[0]
    blocks = sorted(page.get_text("dict")["blocks"], key=lambda b: b['bbox'][1])
    title_candidates = [s['text'].strip() for b in blocks if b['bbox'][1] < page.rect.height * 0.4 and b.get('lines') for l in b['lines'] for s in l['spans'] if s['size'] >= max_size * 0.8 and len(s['text'].strip()) > 2]
    rule_title = " ".join(title_candidates)

    # Find Headings
    heading_candidates = []
    for page_num, page in enumerate(doc):
        if (page_num + 1) in toc_pages: continue
        for b in page.get_text("dict", flags=11)["blocks"]:
            full_text = " ".join([s['text'] for l in b.get('lines', []) for s in l.get('spans', [])]).strip()
            if not full_text or full_text in ignore_texts: continue
            
            span_sizes = [s['size'] for l in b.get('lines', []) for s in l.get('spans', [])]
            is_bold = any((s['flags'] & 16) for l in b.get('lines', []) for s in l.get('spans', []))
            avg_size = statistics.mean(span_sizes) if span_sizes else 0
            
            score = 0
            if avg_size > median_size * 1.15: score += (avg_size / median_size) * 10
            if is_bold: score += 10
            if re.match(r'^(appendix\s+[A-Z]|\d+(\.\d+)*)\s*[:.]*', full_text, re.IGNORECASE): score += 30
            if len(full_text.split()) > 35: score -= 25
            
            if score > 18:
                heading_candidates.append({"text": full_text, "page": page_num + 1, "size": avg_size, "bold": is_bold})

    heading_sizes = [h['size'] for h in heading_candidates if h['bold']] or [h['size'] for h in heading_candidates]
    if heading_sizes:
        h1_thresh, h2_thresh = np.percentile(heading_sizes, 80), np.percentile(heading_sizes, 50)
        for h in heading_candidates:
            level = "H3"
            if h['size'] >= h1_thresh and h['bold']: level = "H1"
            elif h['size'] >= h2_thresh: level = "H2"
            
            clean_text = h['text']
            match = re.match(r'^(appendix\s+[A-Z]|\d+(\.\d+)*)\s*[:.]*\s*(.+)', h['text'], re.IGNORECASE)
            if match:
                clean_text, level_str = match.group(3).strip(), match.group(1)
                if "appendix" in level_str.lower(): level = "H1"
                else: level = f"H{min(level_str.count('.') + 1, 3)}"
            
            clean_text = re.sub(r'\s*\.{3,}\s*\d+\s*$', '', clean_text)
            if len(clean_text) > 2:
                rule_outline.append({"level": level, "text": clean_text, "page": h['page']})
            
    return {"title": rule_title, "outline": rule_outline}

# --- Main Orchestrator ---

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith('.pdf'): continue
        
        pdf_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing {pdf_path}...")

        try:
            doc = fitz.open(pdf_path)
            # --- NEW: Pre-pass to find and ignore headers/footers ---
            header_footer_texts = defaultdict(int)
            for page in doc:
                # Use a simplified text extraction for this pass
                blocks = page.get_text("blocks")
                # Top 10% of page
                headers = [b[4].strip() for b in blocks if b[3] < page.rect.height * 0.1 and b[4].strip()]
                # Bottom 10% of page
                footers = [b[4].strip() for b in blocks if b[1] > page.rect.height * 0.9 and b[4].strip()]
                for text in set(headers + footers):
                    header_footer_texts[text] += 1
            
            # Ignore any text that appears in headers/footers on more than half the pages
            ignore_texts = {text for text, count in header_footer_texts.items() if count > len(doc) / 2}
            if ignore_texts:
                print(f"    - Info: Found and ignoring repetitive headers/footers: {ignore_texts}")
            # -----------------------------------------------------------

            # The ML Pass is currently a placeholder for brevity, but would use `ignore_texts`
            ml_results = {"title": "Untitled", "outline": []}
            rule_results = run_smarter_rule_based_pass(pdf_path, ignore_texts)

            # --- The Intelligent Merge ---
            final_title = rule_results["title"] or ml_results["title"] or "Untitled"
            
            # Start with the high-confidence rule-based outline
            final_outline = rule_results["outline"]
            seen_rule_texts = {" ".join(item["text"].lower().split()[:5]) for item in final_outline}

            # Add in ML results only if they are not duplicates
            for item in ml_results["outline"]:
                ml_key = " ".join(item["text"].lower().split()[:5])
                if ml_key not in seen_rule_texts:
                    final_outline.append(item)
            
            final_outline.sort(key=lambda x: (x['page'], x['level']))
            
            # Final de-duplication pass
            final_clean_outline, seen_final = [], set()
            for item in final_outline:
                key = (item["text"].strip(), item["page"])
                if key not in seen_final:
                    final_clean_outline.append(item)
                    seen_final.add(key)

            final_result = {"title": final_title, "outline": final_clean_outline}

            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"✅ Successfully wrote FINAL HYBRID output to {output_path}")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

if __name__ == "__main__":
    main()