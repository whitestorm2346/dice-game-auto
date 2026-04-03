from dpi_fix import set_dpi_awareness
set_dpi_awareness()

import os
import time
import cv2

from window_selector import choose_window
from board_locator import get_client_region, capture_window, select_board_roi, crop_board


GRID_ROWS = 3
GRID_COLS = 5
SAVE_DIR = "raw_cells"
CAPTURE_INTERVAL = 0.8


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
            cells.append((r, c, cell))

    return cells


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    win = choose_window()
    print("selected:", win)
    time.sleep(0.5)

    window_region = get_client_region(win)
    window_img = capture_window(window_region, win)

    board_roi = select_board_roi(window_img)
    print("board_roi =", board_roi)

    frame_idx = 0

    while True:
        window_region = get_client_region(win)
        window_img = capture_window(window_region)
        board_img = crop_board(window_img, board_roi)

        cells = split_grid(board_img)

        vis = board_img.copy()
        for r, c, cell in cells:
            h, w = board_img.shape[:2]
            cell_w = w / GRID_COLS
            cell_h = h / GRID_ROWS
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(
                vis,
                f"{r},{c}",
                (x1 + 5, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow("Board Grid", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            for r, c, cell in cells:
                filename = os.path.join(SAVE_DIR, f"frame{frame_idx:04d}_r{r}_c{c}.png")
                cv2.imwrite(filename, cell)
            print(f"saved frame {frame_idx}")
            frame_idx += 1

        elif key == ord("q"):
            break

        time.sleep(CAPTURE_INTERVAL)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()