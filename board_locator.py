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

    left, top, right, bottom = win32gui.GetClientRect(hwnd)

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


def _set_window_topmost(hwnd, enabled=True):
    flags = (
        win32con.SWP_NOMOVE
        | win32con.SWP_NOSIZE
        | win32con.SWP_NOACTIVATE
    )

    try:
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST if enabled else win32con.HWND_NOTOPMOST,
            0, 0, 0, 0,
            flags
        )
    except Exception as e:
        action = "啟用最上層" if enabled else "解除最上層"
        print(f"{action}失敗:", repr(e))
        traceback.print_exc()


def _set_opencv_window_topmost(window_name, enabled=True):
    """
    把 OpenCV 視窗設為最上層 / 取消最上層。
    需要先 namedWindow + imshow 之後，視窗句柄才比較拿得到。
    """
    try:
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd != 0:
            _set_window_topmost(hwnd, enabled)
    except Exception as e:
        print("設定 OpenCV 視窗最上層失敗:", repr(e))


def capture_window(window_region, win_info=None, force_visible=True, topmost_only_during_capture=True):
    """
    用 mss 擷取指定區域。
    如果提供 win_info，則可在截圖前短暫把模擬器拉到最上層，
    截完後立刻解除最上層，避免後續 ROI 視窗被蓋住。
    """
    hwnd = None

    if win_info is not None:
        hwnd = win_info["hwnd"]

        if force_visible:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            except Exception as e:
                print("ShowWindow 失敗:", repr(e))

        if topmost_only_during_capture:
            _set_window_topmost(hwnd, True)
            time.sleep(0.03)

    with mss.mss() as sct:
        img = np.array(sct.grab(window_region))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if hwnd is not None and topmost_only_during_capture:
        _set_window_topmost(hwnd, False)

    return img


def select_board_roi(window_img, window_region=None):
    """
    顯示 ROI 選取視窗。
    若提供 window_region，則讓 ROI 視窗的位置與大小貼齊模擬器 client area。
    並且在模擬器解除最上層後，把 ROI 視窗設成最上層。
    """
    window_name = "Board ROI Preview"

    h, w = window_img.shape[:2]

    print("請框選骰子盤面")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, window_img)

    if window_region is not None:
        cv2.resizeWindow(window_name, window_region["width"], window_region["height"])
        cv2.moveWindow(window_name, window_region["left"], window_region["top"])
    else:
        cv2.resizeWindow(window_name, w, h)

    # 讓 OpenCV 視窗真的建立完成
    cv2.waitKey(100)

    # ROI 視窗設為最上層，避免被模擬器或其他視窗蓋住
    _set_opencv_window_topmost(window_name, True)

    # 再等一下，確保置頂生效
    cv2.waitKey(50)

    roi = cv2.selectROI(window_name, window_img, showCrosshair=True, fromCenter=False)

    # 關閉前可順手解除最上層
    _set_opencv_window_topmost(window_name, False)
    cv2.destroyWindow(window_name)

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        raise RuntimeError("ROI 選取失敗")

    return (x, y, rw, rh)


def crop_board(window_img, board_roi):
    x, y, w, h = board_roi
    return window_img[y:y+h, x:x+w]