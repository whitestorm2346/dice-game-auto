import cv2

GRID_ROWS = 3
GRID_COLS = 5

# 你可以視情況調整
CONF_THRESHOLDS = {
    "boulder_ready": 0.80,
    "boulder_charging": 0.80,
    "pink_target": 0.80,
    "other": 0.00,
}


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


def classify_cell_with_cnn(cell_img, predictor):
    pred_label, confidence, score_map = predictor.predict_bgr(cell_img)

    threshold = CONF_THRESHOLDS.get(pred_label, 0.8)

    if pred_label != "other" and confidence < threshold:
        return "other", confidence, score_map

    return pred_label, confidence, score_map


def find_dice(board_img, predictor):
    cells = split_grid(board_img)

    boulder_ready = []
    boulder_charging = []
    pink_target = []

    for cell in cells:
        label, confidence, score_map = classify_cell_with_cnn(cell["img"], predictor)

        cell["label"] = label
        cell["confidence"] = confidence
        cell["scores"] = score_map

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
        confidence = cell.get("confidence", 0.0)
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
        conf_text = f"{confidence:.2f}"

        short_scores = (
            f"R{scores.get('boulder_ready', 0):.2f} "
            f"C{scores.get('boulder_charging', 0):.2f} "
            f"P{scores.get('pink_target', 0):.2f} "
            f"O{scores.get('other', 0):.2f}"
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
            conf_text,
            (x1 + 3, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            vis,
            short_scores,
            (x1 + 3, y1 + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
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