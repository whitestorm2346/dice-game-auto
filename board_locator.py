import time
import traceback
import win32gui
import win32con
import mss
import cv2
import numpy as np


def get_client_region(win_info):
    """
    用 client area 而不是整個 window frame。
    這樣比較接近模擬器真正內容區，而且跨 DPI 比較穩。
    """
    hwnd = win_info["hwnd"]

    # client rect 是相對於 client 原點
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    # 轉成螢幕座標
    screen_left_top = win32gui.ClientToScreen(hwnd, (left, top))
    screen_right_bottom = win32gui.ClientToScreen(hwnd, (right, bottom))

    screen_left, screen_top = screen_left_top
    screen_right, screen_bottom = screen_right_bottom

    return {
        "left": screen_left,
        "top": screen_top,
        "width": screen_right - screen_left,
        "height": screen_bottom - screen_top,
    }


def capture_window(window_region, win_info=None):
    if win_info is not None:
        hwnd = win_info["hwnd"]

        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except Exception as e:
            print("ShowWindow 失敗:", repr(e))

        try:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
            )
            print("SetWindowPos 成功")
        except Exception as e:
            print("SetWindowPos 失敗:", repr(e))
            traceback.print_exc()

        time.sleep(0.3)

    with mss.mss() as sct:
        img = np.array(sct.grab(window_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def select_board_roi(window_img):
    window_name = "Board ROI Preview"

    h, w = window_img.shape[:2]

    print("請在跳出的 OpenCV 視窗中框選骰子盤面。")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, window_img)
    cv2.resizeWindow(window_name, w, h)
    cv2.waitKey(200)

    roi = cv2.selectROI(window_name, window_img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("ROI 選取失敗")

    return (x, y, w, h)


def crop_board(window_img, board_roi):
    x, y, w, h = board_roi
    return window_img[y:y+h, x:x+w]