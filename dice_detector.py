import cv2
import numpy as np


GRID_ROWS = 3
GRID_COLS = 5


def split_grid(board_img, rows=GRID_ROWS, cols=GRID_COLS):
    h, w = board_img.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    cells = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            cell = board_img[y1:y2, x1:x2]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            cells.append({
                "row": r,
                "col": c,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center": (center_x, center_y),
                "img": cell,
            })
    return cells


def crop_center(cell_img, ratio=0.5):
    h, w = cell_img.shape[:2]
    nw = int(w * ratio)
    nh = int(h * ratio)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    return cell_img[y1:y1+nh, x1:x1+nw]


def classify_cell(cell_img):
    """
    回傳:
    - cyan_ready
    - cyan_cooldown
    - pink
    - other
    """
    roi = crop_center(cell_img, ratio=0.55)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 淺藍
    cyan_mask = cv2.inRange(
        hsv,
        np.array([80, 70, 100]),
        np.array([105, 255, 255])
    )

    # 粉紅
    pink_mask = cv2.inRange(
        hsv,
        np.array([140, 60, 100]),
        np.array([179, 255, 255])
    )

    cyan_ratio = np.count_nonzero(cyan_mask) / cyan_mask.size
    pink_ratio = np.count_nonzero(pink_mask) / pink_mask.size

    mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
    _, _, v = mean_hsv

    if pink_ratio > 0.10:
        return "pink"

    if cyan_ratio > 0.12:
        if v > 120:
            return "cyan_ready"
        else:
            return "cyan_cooldown"

    return "other"


def find_dice(board_img):
    cells = split_grid(board_img)

    cyan_ready = []
    pink = []

    for cell in cells:
        label = classify_cell(cell["img"])
        cell["label"] = label

        if label == "cyan_ready":
            cyan_ready.append(cell)
        elif label == "pink":
            pink.append(cell)

    return cyan_ready, pink, cells


def draw_debug(board_img, cells):
    vis = board_img.copy()

    for cell in cells:
        x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
        label = cell.get("label", "other")

        if label == "cyan_ready":
            color = (255, 255, 0)
        elif label == "cyan_cooldown":
            color = (180, 120, 0)
        elif label == "pink":
            color = (255, 0, 255)
        else:
            color = (150, 150, 150)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"{cell['row']},{cell['col']} {label}",
            (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    return vis


def board_to_screen(cell_center, board_roi, window_region):
    roi_x, roi_y, _, _ = board_roi
    cx, cy = cell_center

    screen_x = window_region["left"] + roi_x + cx
    screen_y = window_region["top"] + roi_y + cy
    return (screen_x, screen_y)