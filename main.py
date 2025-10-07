# main.py
import os
import random
from anotation import (
    GroundingDINOHFInference,
    prepare_coco_annotations,
    COCOHandler
)

def main_process(category='dog', output_dir='outputs/dog', num_images=300):
    # --- 1. COCOデータセット ダウンロード設定 ---
    COCO_DATA_DIR = 'coco_data'
    COCO_CATEGORY = category
    NUM_IMAGES = num_images

    # --- 2. YOLOアノテーション作成設定 ---
    OUTPUT_DIR = os.path.join(output_dir, 'yolo_labels')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CLASS_MAPPING = {COCO_CATEGORY: 0}
    PROMPT = f"{COCO_CATEGORY}."

    # --- 手順A: COCOアノテーションファイルの準備 ---
    print(f"--- {COCO_CATEGORY}: COCOアノテーションファイルの準備 ---")
    ann_paths = prepare_coco_annotations(data_dir=COCO_DATA_DIR, extract_val=True)
    val_ann_path = ann_paths.get('val_json')

    if not val_ann_path:
        print("エラー: COCOの検証用アノテーションファイルが見つかりません。")
        return

    # --- 手順B: COCO画像ダウンロード ---
    print(f"--- {COCO_CATEGORY}: {NUM_IMAGES}枚の画像をダウンロード ---")
    coco = COCOHandler(data_dir=COCO_DATA_DIR, annotation_path=val_ann_path, dataset_type='val2017')
    img_ids = coco.get_image_ids(category_name=COCO_CATEGORY)
    if not img_ids:
        print(f"カテゴリ '{COCO_CATEGORY}' の画像が見つかりません。")
        return

    selected_ids = random.sample(img_ids, min(NUM_IMAGES, len(img_ids)))
    for img_id in selected_ids:
        img_info = coco.coco.loadImgs(img_id)[0]
        coco.download_image(img_info)

    IMAGE_DIR = coco.images_dir
    print(f"--- {COCO_CATEGORY}: 画像ダウンロード完了 ({IMAGE_DIR}) ---")

    # --- 手順C: YOLOデータセット作成 ---
    print(f"--- {COCO_CATEGORY}: GroundingDINOでYOLOデータセット作成 ---")
    grounding_dino = GroundingDINOHFInference()
    grounding_dino.create_yolo_dataset(
        image_folder_path=IMAGE_DIR,
        output_folder_path=OUTPUT_DIR,
        class_mapping=CLASS_MAPPING,
        prompt=PROMPT
    )

    print(f"--- {COCO_CATEGORY}: YOLOラベル生成完了 ---")
    print("--- 全ての処理が完了しました ---")


if __name__ == "__main__":
    # 手動実行用
    main_process(category='dog', output_dir='outputs/dog', num_images=300)
