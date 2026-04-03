import os
import cv2
import numpy as np


GRID_ROWS = 3
GRID_COLS = 5

CLASS_NAMES = [
    "boulder_ready",
    "boulder_charging",
    "pink_target",
]

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


def crop_center(cell_img, ratio=0.85):
    h, w = cell_img.shape[:2]
    nw = int(w * ratio)
    nh = int(h * ratio)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    return cell_img[y1:y1+nh, x1:x1+nw]


def preprocess(img, size=(64, 64)):
    """
    只保留中央區域，再 resize，最後轉灰階
    """
    roi = crop_center(img, ratio=0.72)
    roi = cv2.resize(roi, size)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray


def load_templates():
    templates = {class_name: [] for class_name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        folder = os.path.join("templates", class_name)

        if not os.path.isdir(folder):
            raise RuntimeError(f"找不到模板資料夾: {folder}")

        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            templates[class_name].append(preprocess(img))

        if not templates[class_name]:
            raise RuntimeError(f"{class_name} 沒有任何可用模板")

    return templates

def match_one_template(cell_img, template_img):
    cell_proc = preprocess(cell_img)

    result = cv2.matchTemplate(cell_proc, template_img, cv2.TM_CCOEFF_NORMED)
    score = float(result[0][0])
    return score


def classify_cell(cell_img, templates, thresholds=None):
    if thresholds is None:
        thresholds = {
            "boulder_ready": 0.50,
            "boulder_charging": 0.50,
            "pink_target": 0.50,
        }

    scores = {}

    for class_name, template_list in templates.items():
        best_score = -1.0

        for tmpl in template_list:
            score = match_one_template(cell_img, tmpl)
            if score > best_score:
                best_score = score

        scores[class_name] = best_score

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_score < thresholds.get(best_label, 0.50):
        return "other", scores

    return best_label, scores


def find_dice(board_img, templates):
    cells = split_grid(board_img)

    boulder_ready = []
    boulder_charging = []
    pink_target = []

    for cell in cells:
        label, scores = classify_cell(cell["img"], templates)
        cell["label"] = label
        cell["scores"] = scores

        if label == "boulder_ready":
            boulder_ready.append(cell)
        elif label == "boulder_charging":
            boulder_charging.append(cell)
        elif label == "pink_target":
            pink_target.append(cell)

    return boulder_ready, boulder_charging, pink_target, cells


def draw_debug(board_img, cells):
    vis = board_img.copy()

    for cell in cells:
        x1, y1, x2, y2 = cell["x1"], cell["y1"], cell["x2"], cell["y2"]
        label = cell.get("label", "other")
        scores = cell.get("scores", {})

        if label == "boulder_ready":
            color = (0, 220, 255)
        elif label == "boulder_charging":
            color = (180, 255, 255)
        elif label == "pink_target":
            color = (255, 0, 255)
        else:
            color = (120, 120, 120)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        main_text = f"{cell['row']},{cell['col']} {label}"
        score_text = (
            f"R{scores.get('boulder_ready', 0):.2f} "
            f"C{scores.get('boulder_charging', 0):.2f} "
            f"P{scores.get('pink_target', 0):.2f}"
        )

        cv2.putText(
            vis,
            main_text,
            (x1 + 3, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            score_text,
            (x1 + 3, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
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