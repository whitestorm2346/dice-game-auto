import os
import shutil
import cv2


RAW_DIR = "raw_cells"
OUTPUT_BASE = "dataset/train"

LABEL_MAP = {
    ord("1"): "boulder_ready",
    ord("2"): "boulder_charging",
    ord("3"): "pink_target",
    ord("0"): "other",
}


def ensure_dirs():
    for class_name in LABEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_BASE, class_name), exist_ok=True)


def main():
    ensure_dirs()

    files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    files.sort()

    if not files:
        print("raw_cells 裡沒有圖片")
        return

    idx = 0

    while idx < len(files):
        filename = files[idx]
        path = os.path.join(RAW_DIR, filename)

        img = cv2.imread(path)
        if img is None:
            print("讀取失敗，跳過:", filename)
            idx += 1
            continue

        display = img.copy()
        h, w = display.shape[:2]
        scale = min(300 / w, 300 / h, 1.0)
        show = cv2.resize(display, (int(w * scale), int(h * scale)))

        info = (
            f"{idx+1}/{len(files)} | "
            f"[1] ready  [2] charging  [3] pink  [0] other  [q] quit"
        )
        cv2.putText(
            show,
            info,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Label Raw Cells", show)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        if key in LABEL_MAP:
            class_name = LABEL_MAP[key]
            dst_path = os.path.join(OUTPUT_BASE, class_name, filename)
            shutil.move(path, dst_path)
            print(f"{filename} -> {class_name}")
            idx += 1
        else:
            print("無效按鍵，請用 1 / 2 / 3 / 0 / q")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()