import os
import yaml
from main import main_process  # 関数化したmain.py
from ultralytics import YOLO
import shutil, glob, random

# --- COCO80クラスリスト ---
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- 共通学習パラメータ ---
EPOCHS = 50
IMGSZ = 640
BATCH = 16

# --- 自動化ループ ---
for cls in COCO_CLASSES:
    print(f"\n===== 処理開始: {cls} =====")
    
    base_dir = f"/content/outputs/{cls}"
    os.makedirs(base_dir, exist_ok=True)

    # --- Step1: 画像ダウンロード & YOLOラベル生成 ---
    main_process(category=cls, output_dir=base_dir, num_images=300)

    # --- Step2: データ分割 ---
    img_folder = os.path.join(base_dir, "yolo_labels")
    train_val_base = os.path.join(base_dir, "mydata")
    train_img_dir = os.path.join(train_val_base, "images/train")
    val_img_dir   = os.path.join(train_val_base, "images/val")
    train_lbl_dir = os.path.join(train_val_base, "labels/train")
    val_lbl_dir   = os.path.join(train_val_base, "labels/val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # YOLO形式ラベルと画像を取得
    images = glob.glob(os.path.join(img_folder, "*.jpg"))
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # コピー関数
    def copy_dataset(img_list, dst_img_dir, dst_lbl_dir):
        for img_path in img_list:
            base = os.path.basename(img_path)
            lbl_path = os.path.join(img_folder, base.replace(".jpg", ".txt"))
            shutil.copy(img_path, dst_img_dir)
            shutil.copy(lbl_path, dst_lbl_dir)

    copy_dataset(train_images, train_img_dir, train_lbl_dir)
    copy_dataset(val_images, val_img_dir, val_lbl_dir)
    print(f"{cls}のtrain/val分割とコピー完了")

    # --- Step3: data.yaml 自動生成 ---
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    data = {
        'train': train_img_dir,
        'val': val_img_dir,
        'nc': 1,
        'names': {0: cls}
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"{cls} の data.yaml を生成しました: {data_yaml_path}")

    # --- Step4: YOLOv8学習 ---
    print(f"{cls} のYOLOv8学習開始")
    model = YOLO("yolov8m.pt")
    model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=os.path.join(base_dir, "runs_train"),
        name=f"model_{cls}"
    )
    print(f"{cls} の学習完了\n")
