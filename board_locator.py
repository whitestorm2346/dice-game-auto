import cv2
import mss
import numpy as np


def capture_region(region):
    with mss.mss() as sct:
        img = np.array(sct.grab(region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def capture_window(window_region):
    return capture_region(window_region)


def select_board_roi(window_img):
    print("請框選骰子盤面區域，按 Enter 確認，按 c 取消")
    roi = cv2.selectROI("Select Dice Board ROI", window_img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Dice Board ROI")

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("ROI 選取失敗")

    return (x, y, w, h)


def crop_board(window_img, board_roi):
    x, y, w, h = board_roi
    return window_img[y:y+h, x:x+w]