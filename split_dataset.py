import os
import random
import shutil
import argparse


TRAIN_BASE = "dataset/train"
VAL_BASE = "dataset/val"
VAL_RATIO = 0.2

CLASS_NAMES = [
    "boulder_ready",
    "boulder_charging",
    "pink_target",
    "other",
]


def move_all_val_back_to_train():
    print("=== 將 val 全部移回 train ===")

    for class_name in CLASS_NAMES:
        val_dir = os.path.join(VAL_BASE, class_name)
        train_dir = os.path.join(TRAIN_BASE, class_name)

        if not os.path.exists(val_dir):
            continue

        files = os.listdir(val_dir)

        moved = 0
        for f in files:
            src = os.path.join(val_dir, f)
            dst = os.path.join(train_dir, f)
            shutil.move(src, dst)
            moved += 1

        print(f"{class_name}: moved {moved} files back to train")


def split_train_to_val():
    print("=== 重新切 train → val ===")

    random.seed(42)

    for class_name in CLASS_NAMES:
        train_dir = os.path.join(TRAIN_BASE, class_name)
        val_dir = os.path.join(VAL_BASE, class_name)

        os.makedirs(val_dir, exist_ok=True)

        files = [
            f for f in os.listdir(train_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        random.shuffle(files)

        n_val = int(len(files) * VAL_RATIO)
        val_files = files[:n_val]

        for f in val_files:
            src = os.path.join(train_dir, f)
            dst = os.path.join(val_dir, f)
            shutil.move(src, dst)

        print(f"{class_name}: moved {n_val} files to val")


def main():
    parser = argparse.ArgumentParser(description="Dataset split tool")

    # 👇 同時支援長參數與短參數
    parser.add_argument("-r", "--reset", action="store_true", help="把 val → train")
    parser.add_argument("-s", "--split", action="store_true", help="train → val")

    args = parser.parse_args()

    if args.reset:
        move_all_val_back_to_train()

    if args.split:
        split_train_to_val()

    if not args.reset and not args.split:
        print("請使用參數:")
        print("  -r 或 --reset   把 val → train")
        print("  -s 或 --split   重新切 train → val")
        print("範例:")
        print("  python split_dataset.py -r -s")


if __name__ == "__main__":
    main()