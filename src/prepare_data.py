# src/prepare_data.py (CORRECT - Fuses COCO and JSON folders)
import json
import os
from tqdm import tqdm

# --- Configuration ---
DOCLAYNET_PATH = 'data/raw/doclaynet'
OUTPUT_PATH = 'data/processed/unified_data.jsonl'

# --- Label Mapping ---
LABEL_MAP = {
    'Title': 'TITLE', 'Section-header': 'H1', 'Text': 'BODY',
    'Page-footer': 'OTHER', 'Page-header': 'OTHER', 'List-item': 'BODY',
    'Caption': 'OTHER', 'Picture': 'OTHER', 'Table': 'OTHER',
    'Figure': 'OTHER', 'Formula': 'OTHER', 'default': 'OTHER'
}

def normalize_bbox(bbox, width, height):
    """Normalize bbox to a 0-1000 integer scale, and clamp values to be safe."""
    x, y, w, h = bbox
    # Clamp values to ensure they are within the [0, 1000] range
    x0 = max(0, min(int(1000 * x / width), 1000))
    y0 = max(0, min(int(1000 * y / height), 1000))
    x1 = max(0, min(int(1000 * (x + w) / width), 1000))
    y1 = max(0, min(int(1000 * (y + h) / height), 1000))
    return [x0, y0, x1, y1]

def is_text_in_bbox(text_bbox, anno_bbox):
    """Check if the center of a text box is inside an annotation box."""
    tx, ty, tw, th = text_bbox
    ax, ay, aw, ah = anno_bbox
    text_center_x, text_center_y = tx + tw / 2, ty + th / 2
    return (ax <= text_center_x <= (ax + aw)) and (ay <= text_center_y <= (ay + ah))

def process_doclaynet_file(coco_json_path, json_dir_path):
    """Processes a COCO file by fusing it with page-level JSONs."""
    print(f"Processing {os.path.basename(coco_json_path)}...")
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_id_to_info = {img['id']: img for img in data['images']}
    image_id_to_annos = {}
    for anno in data['annotations']:
        img_id = anno['image_id']
        if img_id not in image_id_to_annos: image_id_to_annos[img_id] = []
        image_id_to_annos[img_id].append(anno)
    
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    processed_pages = []
    for image_id, annos in tqdm(image_id_to_annos.items(), desc="Fusing data for pages"):
        image_info = image_id_to_info.get(image_id)
        if not image_info: continue

        width, height = image_info['width'], image_info['height']
        page_json_filename = os.path.splitext(image_info['file_name'])[0] + '.json'
        page_json_path = os.path.join(json_dir_path, page_json_filename)

        if not os.path.exists(page_json_path): continue
            
        with open(page_json_path, 'r', encoding='utf-8') as f:
            page_text_data = json.load(f)

        page_elements = []
        for anno in annos:
            anno_bbox = anno['bbox']
            label_name = category_id_to_name.get(anno['category_id'], 'default')
            final_label = LABEL_MAP.get(label_name, 'OTHER')

            contained_texts = []
            for text_cell in page_text_data['cells']:
                if is_text_in_bbox(text_cell['bbox'], anno_bbox):
                    contained_texts.append((text_cell['text'], text_cell['bbox'][1]))
            
            contained_texts.sort(key=lambda x: x[1])
            full_text = " ".join([text for text, y in contained_texts]).strip()

            if full_text:
                page_elements.append({
                    'text': full_text, 'label': final_label,
                    'bbox': normalize_bbox(anno_bbox, width, height)
                })

        if page_elements:
            words, labels, bboxes = [], [], []
            for elem in page_elements:
                for word in elem['text'].split():
                    if word.strip():
                        words.append(word)
                        labels.append(elem['label'])
                        bboxes.append(elem['bbox'])
            
            if words:
                processed_pages.append({
                    'id': image_info['file_name'], 'words': words,
                    'labels': labels, 'bboxes': bboxes
                })
            
    return processed_pages

def main():
    coco_train_json = os.path.join(DOCLAYNET_PATH, 'COCO', 'train.json')
    coco_val_json = os.path.join(DOCLAYNET_PATH, 'COCO', 'val.json')
    page_json_dir = os.path.join(DOCLAYNET_PATH, 'JSON')

    if not os.path.exists(page_json_dir):
        print("\n❌ FATAL ERROR: The 'JSON' folder was not found in 'data/raw/doclaynet'.")
        print("Please ensure it has been copied correctly.")
        return

    all_pages = []
    if os.path.exists(coco_train_json):
        all_pages.extend(process_doclaynet_file(coco_train_json, page_json_dir))
    if os.path.exists(coco_val_json):
        all_pages.extend(process_doclaynet_file(coco_val_json, page_json_dir))
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for page in all_pages:
            f.write(json.dumps(page) + '\n')
            
    print(f"\n✅ Successfully processed {len(all_pages)} pages.")
    print(f"Unified data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()