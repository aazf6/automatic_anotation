# /content/automatic_anotation/auto_main.py

import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from main import main_process

# === COCO80カテゴリリスト ===
COCO_CATEGORIES = [
    'person'
]

# === 各カテゴリ処理 ===
for category in COCO_CATEGORIES:
    print(f"\n========== {category.upper()} の処理を開始 ==========")
    output_dir = f"/content/automatic_anotation/outputs/{category}"

    # --- 1. main.pyの処理を実行（画像DL + YOLOラベル作成） ---
    main_process(category=category, output_dir=output_dir, num_images=300)

    # --- 2. train/val分割＆コピー ---
    print(f"--- {category}: データ分割を実行 ---")
    images = sorted(glob.glob("/content/automatic_anotation/coco_data/val2017/*.jpg"))
    labels = sorted(glob.glob(f"{output_dir}/yolo_labels/*.txt"))

    if len(images) == 0 or len(labels) == 0:
        print(f"[スキップ] {category} に対応する画像またはラベルが見つかりません。")
        continue

    # train/val分割
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # mydata作成
    base_dir = f"/content/automatic_anotation/mydata/{category}"
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    # trainコピー
    for img, lbl in zip(train_imgs, train_labels):
        shutil.copy(img, os.path.join(base_dir, "images/train"))
        shutil.copy(lbl, os.path.join(base_dir, "labels/train"))

    # valコピー
    for img, lbl in zip(val_imgs, val_labels):
        shutil.copy(img, os.path.join(base_dir, "images/val"))
        shutil.copy(lbl, os.path.join(base_dir, "labels/val"))

    print(f"--- {category}: train/val分割＆コピー完了 ---")

    # --- 3. data.yaml生成 ---
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""train: {base_dir}/images/train
val: {base_dir}/images/val

nc: 1
names:
  0: {category}
""")
    print(f"--- {category}: data.yamlを生成しました ({yaml_path}) ---")

print("\n✅ すべてのカテゴリ処理が完了しました！")

from ultralytics import YOLO

# YOLOモデルのロード
model = YOLO("yolov8m.pt")

# COCOクラスのループ（上でmain_processを回したあと）
for category in COCO_CATEGORIES:
    yaml_path = f"/content/automatic_anotation/outputs/{category}/data.yaml"
    if not os.path.exists(yaml_path):
        print(f"[スキップ] {category}: data.yamlが存在しません。")
        continue

    print(f"\n🚀 {category} モデルの学習を開始します ---")
    model.train(
        data=yaml_path,
        epochs=100,
        batch=16,
        imgsz=640
    )

    # 学習済みモデルを保存（任意）
    save_dir = f"/content/automatic_anotation/outputs/{category}/trained"
    os.makedirs(save_dir, exist_ok=True)
    model.export(format="pt", project=save_dir, name=f"{category}_best")

    print(f"✅ {category}: 学習完了 & モデル保存 ({save_dir})")
