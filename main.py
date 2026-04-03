from dpi_fix import set_dpi_awareness
set_dpi_awareness()

import time
import cv2
import keyboard  # pip install keyboard

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

ready_pick_index = 0
pink_pick_index = 0

# ===== 全域控制狀態 =====
running = True
paused = False
manual_drag_requested = False
roi_select_requested = False


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


# ===== 全域熱鍵函式 =====
def request_roi_selection():
    global roi_select_requested
    roi_select_requested = True
    print("[HOTKEY] ROI selection requested")


def toggle_pause():
    global paused
    paused = not paused
    print(f"[HOTKEY] paused = {paused}")


def stop_program():
    global running
    running = False
    print("[HOTKEY] stopping program...")


def toggle_auto_drag():
    global AUTO_DRAG
    AUTO_DRAG = not AUTO_DRAG
    print(f"[HOTKEY] AUTO_DRAG = {AUTO_DRAG}")


def request_manual_drag():
    global manual_drag_requested
    manual_drag_requested = True
    print("[HOTKEY] manual drag requested")


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

    return True


def main():
    global running, paused, manual_drag_requested, roi_select_requested

    print("=== Dice Bot (CNN Live Test) ===")

    # 先註冊全域熱鍵
    register_hotkeys()

    # 1. 選視窗
    win = choose_window()
    print("selected:", win)
    time.sleep(0.5)

    # 2. 載入 CNN 模型
    predictor = DicePredictor("dice_cnn.pth")
    print("CNN 模型載入成功")

    # 3. 啟動時先不選 ROI，等你按 F7 再選
    board_roi = None
    print("尚未選取盤面 ROI，請先把模擬器準備好，再按 F7 進行格子區選取")

    last_drag_time = 0
    last_status_print_time = 0

    while running:
        try:
            # 優先處理 ROI 選取請求
            if roi_select_requested:
                roi_select_requested = False

                window_region = get_client_region(win)
                window_img = capture_window(window_region, win)
                board_roi = select_board_roi(window_img, window_region)
                print(f"盤面 ROI = {board_roi}")

                # 選完 ROI 後關掉 ROI 視窗
                cv2.destroyAllWindows()
                continue

            if paused:
                cv2.waitKey(1)
                time.sleep(PAUSE_SLEEP)
                continue

            # 還沒選 ROI 前，不進入掃描邏輯
            if board_roi is None:
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
                    print("[MANUAL] 沒有找到可拖曳的巨石骰或粉色目標骰")

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
                print("[cv2] q pressed -> stop")
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
            print("錯誤:", e)
            time.sleep(0.3)

    try:
        keyboard.unhook_all()
    except Exception as e:
        print(f"[HOTKEY] cleanup skipped: {e}")

    cv2.destroyAllWindows()
    print("程式已結束")


if __name__ == "__main__":
    main()