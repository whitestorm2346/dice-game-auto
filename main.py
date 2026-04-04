from dpi_fix import set_dpi_awareness
set_dpi_awareness()

import os
import sys
import time
import cv2
import keyboard
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from window_selector import choose_window
from board_locator import get_client_region, capture_window, select_board_roi, crop_board
from dice_detector import find_dice, draw_debug, board_to_screen
from predict_classifier import DicePredictor
from mouse_controller import drag_dice


# ===== 效能 / 行為設定 =====
SCAN_INTERVAL = 0.005
DRAG_COOLDOWN = 0.0
ACTION_SETTLE_TIME = 0.01
AUTO_DRAG = True

SHOW_DEBUG_WINDOW = False       # 是否顯示 Board Debug 視窗
CHECK_DRAG_RESULT = False       # 是否檢查拖曳後盤面是否改變
SHOW_DRAG_LOG = False           # 是否輸出 drag debug 訊息
STATUS_PRINT_INTERVAL = 0.3     # 幾秒印一次狀態
PAUSE_SLEEP = 0.05

SHOW_CONTROL_PANEL = True       # 是否顯示熱鍵 / 狀態面板


ready_pick_index = 0
pink_pick_index = 0

# ===== 全域控制狀態 =====
running = True
paused = False
manual_drag_requested = False
roi_select_requested = False

# 顯示在控制面板上的最近狀態文字
last_status_message = "Program started"

def resource_path(path):
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, path)


def set_status(msg):
    global last_status_message
    last_status_message = msg
    print(msg)


def get_zh_font(font_size=24):
    """
    取得 Windows 常見中文字型。
    """
    font_candidates = [
        r"C:\Windows\Fonts\msjh.ttc",      # 微軟正黑體
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\mingliu.ttc",   # 細明體
        r"C:\Windows\Fonts\kaiu.ttf",      # 標楷體
    ]

    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass

    return ImageFont.load_default()


def put_text_zh(img, text, pos, font_size=24, color=(255, 255, 255)):
    """
    在 OpenCV 圖片上用 Pillow 畫中文。
    color 請用 BGR 傳入，函式內會轉成 RGB。
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_zh_font(font_size)

    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def has_board_changed(before_img, after_img, threshold=3.0):
    diff = cv2.absdiff(before_img, after_img)
    mean_diff = diff.mean()
    print(f"mean_diff={mean_diff:.3f}")
    return mean_diff >= threshold


def choose_any_ready_and_any_pink(
    boulder_ready,
    pink_target,
    ready_index,
    pink_index,
):
    """
    不看距離，直接從整個盤面輪流挑：
    - ready 不要永遠只挑前面幾顆
    - pink 如果有 1~2 顆也輪流用
    """
    if not boulder_ready or not pink_target:
        return None, None, ready_index, pink_index

    src = boulder_ready[ready_index % len(boulder_ready)]
    dst = pink_target[pink_index % len(pink_target)]

    ready_index += 1
    pink_index += 1

    return src, dst, ready_index, pink_index


def render_control_panel(board_roi):
    """
    顯示所有熱鍵與目前狀態。
    英文固定欄位可用 cv2.putText。
    中文說明與最後訊息改用 Pillow，避免亂碼。
    """
    panel_h = 560
    panel_w = 560
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    title_color = (0, 255, 255)
    label_color = (220, 220, 220)
    value_on_color = (0, 255, 0)
    value_off_color = (0, 120, 255)
    info_color = (255, 255, 255)

    y = 30
    line_gap = 32

    cv2.putText(panel, "Dice Bot Control Panel", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, title_color, 2, cv2.LINE_AA)

    y += 42
    cv2.putText(panel, "Hotkeys:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

    hotkey_lines = [
        ("F7",  "框選骰子區域"),
        ("F8",  "暫停 / 繼續"),
        ("F9",  "切換自動拖曳"),
        ("F10", "手動拖曳一次"),
        ("ESC", "結束應用程式"),
    ]

    for key_name, desc in hotkey_lines:
        y += line_gap
        cv2.putText(panel, f"{key_name:<4} :", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, info_color, 1, cv2.LINE_AA)
        panel = put_text_zh(panel, desc, (115, y - 20), font_size=24, color=info_color)

    y += 46
    cv2.putText(panel, "Status:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

    y += line_gap
    cv2.putText(panel, "Running:", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, str(running), (180, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                value_on_color if running else value_off_color, 2, cv2.LINE_AA)

    y += line_gap
    cv2.putText(panel, "Paused:", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, str(paused), (180, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                value_off_color if paused else value_off_color, 2, cv2.LINE_AA)

    y += line_gap
    cv2.putText(panel, "Auto Drag:", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, str(AUTO_DRAG), (180, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                value_on_color if AUTO_DRAG else value_off_color, 2, cv2.LINE_AA)

    y += line_gap
    roi_selected = board_roi is not None
    cv2.putText(panel, "ROI Selected:", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, str(roi_selected), (180, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                value_on_color if roi_selected else value_off_color, 2, cv2.LINE_AA)

    y += line_gap
    cv2.putText(panel, "Debug Window:", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(panel, str(SHOW_DEBUG_WINDOW), (180, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                value_on_color if SHOW_DEBUG_WINDOW else value_off_color, 2, cv2.LINE_AA)

    y += 46
    cv2.putText(panel, "Last Message:", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

    # 中文訊息改用 Pillow 畫，避免亂碼
    message_lines = []
    max_chars_per_line = 22
    msg = last_status_message

    while msg:
        message_lines.append(msg[:max_chars_per_line])
        msg = msg[max_chars_per_line:]
        if len(message_lines) >= 4:
            break

    for line in message_lines:
        y += line_gap
        panel = put_text_zh(panel, line, (25, y - 20), font_size=22, color=info_color)

    cv2.imshow("Dice Bot Control Panel", panel)


# ===== 全域熱鍵函式 =====
def request_roi_selection():
    global roi_select_requested
    roi_select_requested = True
    set_status("[HOTKEY] ROI selection requested")


def toggle_pause():
    global paused
    paused = not paused
    set_status(f"[HOTKEY] paused = {paused}")


def stop_program():
    global running
    running = False
    set_status("[HOTKEY] stopping program...")


def toggle_auto_drag():
    global AUTO_DRAG
    AUTO_DRAG = not AUTO_DRAG
    set_status(f"[HOTKEY] AUTO_DRAG = {AUTO_DRAG}")


def request_manual_drag():
    global manual_drag_requested
    manual_drag_requested = True
    set_status("[HOTKEY] manual drag requested")


def register_hotkeys():
    keyboard.add_hotkey("f7", request_roi_selection)  # 觸發 ROI 選取
    keyboard.add_hotkey("f8", toggle_pause)           # 暫停 / 繼續
    keyboard.add_hotkey("f9", toggle_auto_drag)       # 自動拖曳開關
    keyboard.add_hotkey("f10", request_manual_drag)   # 手動拖一次
    keyboard.add_hotkey("esc", stop_program)          # 結束程式

    print("=== Global Hotkeys ===")
    print("F7  : Select Board ROI")
    print("F8  : Pause / Resume")
    print("F9  : Toggle Auto Drag")
    print("F10 : Manual Drag Once")
    print("ESC : Stop Program")


def do_one_drag(
    win,
    board_roi,
    window_region,
    board_img,
    boulder_ready,
    pink_target,
    drag_mode="AUTO"
):
    global ready_pick_index, pink_pick_index

    if not boulder_ready or not pink_target:
        return False

    src, dst, ready_pick_index, pink_pick_index = choose_any_ready_and_any_pink(
        boulder_ready,
        pink_target,
        ready_pick_index,
        pink_pick_index
    )

    if src is None or dst is None:
        return False

    src_screen = board_to_screen(src["center"], board_roi, window_region)
    dst_screen = board_to_screen(dst["center"], board_roi, window_region)

    if SHOW_DRAG_LOG:
        print(
            f"[{drag_mode}] drag: boulder_ready ({src['row']},{src['col']}) "
            f"-> pink_target ({dst['row']},{dst['col']})"
        )
        print(f"[{drag_mode}] screen: {src_screen} -> {dst_screen}")

    if CHECK_DRAG_RESULT:
        before_board = board_img.copy()

    drag_dice(
        src_screen,
        dst_screen,
        move_duration=0.05 if drag_mode == "AUTO" else 0.08,
        hold_before_drag=0.02 if drag_mode == "AUTO" else 0.03
    )

    if CHECK_DRAG_RESULT:
        time.sleep(0.05)
        window_img_after = capture_window(window_region, win)
        after_board = crop_board(window_img_after, board_roi)
        changed = has_board_changed(before_board, after_board)

        if SHOW_DRAG_LOG:
            print(f"[{drag_mode}] drag success? {'YES' if changed else 'NO'}")

    set_status(
        f"[{drag_mode}] drag ({src['row']},{src['col']}) -> ({dst['row']},{dst['col']})"
    )
    return True


def main():
    global running, paused, manual_drag_requested, roi_select_requested

    print("=== Dice Bot (CNN Live Test) ===")

    register_hotkeys()

    # 1. 選視窗
    win = choose_window()
    print("selected:", win)
    time.sleep(0.5)

    # 2. 載入 CNN 模型
    predictor = DicePredictor(resource_path("dice_cnn.pth"))
    set_status("CNN 模型載入成功")

    # 3. 啟動時先不選 ROI，等你按 F7 再選
    board_roi = None
    set_status("尚未選取盤面 ROI，請先把模擬器準備好，再按 F7")

    last_drag_time = 0
    last_status_print_time = 0

    while running:
        try:
            if SHOW_CONTROL_PANEL:
                render_control_panel(board_roi)

            # 優先處理 ROI 選取請求
            if roi_select_requested:
                roi_select_requested = False

                window_region = get_client_region(win)
                window_img = capture_window(window_region, win)
                board_roi = select_board_roi(window_img, window_region)
                set_status(f"盤面 ROI = {board_roi}")

                try:
                    cv2.destroyWindow("Board ROI Preview")
                except Exception:
                    pass
                continue

            if paused:
                cv2.waitKey(1)
                time.sleep(PAUSE_SLEEP)
                continue

            # 還沒選 ROI 前，不進入掃描邏輯
            if board_roi is None:
                cv2.waitKey(1)
                time.sleep(0.05)
                continue

            window_region = get_client_region(win)
            window_img = capture_window(window_region, win)
            board_img = crop_board(window_img, board_roi)

            # 4. CNN 即時辨識
            boulder_ready, boulder_charging, pink_target, cells = find_dice(board_img, predictor)

            if SHOW_DEBUG_WINDOW:
                debug_img = draw_debug(board_img, cells)
                cv2.imshow("Board Debug", debug_img)

            now = time.time()

            # 降低 console 輸出頻率
            if now - last_status_print_time >= STATUS_PRINT_INTERVAL:
                print(
                    f"boulder_ready={len(boulder_ready)}, "
                    f"boulder_charging={len(boulder_charging)}, "
                    f"pink_target={len(pink_target)}, "
                    f"AUTO_DRAG={AUTO_DRAG}, paused={paused}, "
                    f"roi_selected={board_roi is not None}"
                )
                last_status_print_time = now

            # 5A. 手動熱鍵觸發一次拖曳（優先）
            if manual_drag_requested:
                manual_drag_requested = False

                did_drag = do_one_drag(
                    win=win,
                    board_roi=board_roi,
                    window_region=window_region,
                    board_img=board_img,
                    boulder_ready=boulder_ready,
                    pink_target=pink_target,
                    drag_mode="MANUAL"
                )

                if did_drag:
                    last_drag_time = time.time()
                    time.sleep(ACTION_SETTLE_TIME)
                    continue
                else:
                    set_status("[MANUAL] 沒有找到可拖曳的巨石骰或粉色目標骰")

            # 5B. 自動拖曳：每次只拖一次，拖完立刻重掃整盤
            if AUTO_DRAG and boulder_ready and pink_target and (now - last_drag_time >= DRAG_COOLDOWN):
                did_drag = do_one_drag(
                    win=win,
                    board_roi=board_roi,
                    window_region=window_region,
                    board_img=board_img,
                    boulder_ready=boulder_ready,
                    pink_target=pink_target,
                    drag_mode="AUTO"
                )

                if did_drag:
                    last_drag_time = time.time()
                    time.sleep(ACTION_SETTLE_TIME)
                    continue

            # 保留 OpenCV 視窗按鍵作為備援
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                set_status("[cv2] q pressed -> stop")
                break
            elif key == ord("r"):
                request_roi_selection()
            elif key == ord("p"):
                toggle_pause()
            elif key == ord("a"):
                toggle_auto_drag()
            elif key == ord("d"):
                request_manual_drag()

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            set_status(f"錯誤: {e}")
            time.sleep(0.3)

    try:
        keyboard.unhook_all()
    except Exception as e:
        print(f"[HOTKEY] cleanup skipped: {e}")

    cv2.destroyAllWindows()
    print("程式已結束")


if __name__ == "__main__":
    main()