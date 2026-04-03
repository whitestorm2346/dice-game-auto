from dpi_fix import set_dpi_awareness
set_dpi_awareness()

import time
import cv2

from window_selector import choose_window
from board_locator import get_client_region, capture_window, select_board_roi, crop_board
from dice_detector import load_templates, find_dice, draw_debug, board_to_screen
from mouse_controller import drag_dice


SCAN_INTERVAL = 0.8
DRAG_COOLDOWN = 1.0
AUTO_DRAG = False   # 先關掉，自行按 d 測試


def has_board_changed(before_img, after_img, threshold=3.0):
    diff = cv2.absdiff(before_img, after_img)
    mean_diff = diff.mean()
    print(f"mean_diff={mean_diff:.3f}")
    return mean_diff >= threshold


def main():
    print("=== Dice Bot ===")

    # 1. 選視窗
    win = choose_window()
    print("selected:", win)
    time.sleep(0.5)

    # 3. 擷取視窗，手動框選盤面
    window_region = get_client_region(win)
    print("region:", window_region)
    window_img = capture_window(window_region, win)

    board_roi = select_board_roi(window_img)
    print(f"盤面 ROI = {board_roi}")

    last_drag_time = 0

    while True:
        try:
            # 每次都重抓，避免使用者移動模擬器視窗
            window_region = get_client_region(win)
            window_img = capture_window(window_region, win)
            board_img = crop_board(window_img, board_roi)

            # 4. 骰子辨識
            boulder_ready, boulder_charging, pink_target, cells = find_dice(board_img, templates)

            debug_img = draw_debug(board_img, cells)
            cv2.imshow("Board Debug", debug_img)

            print(
                f"boulder_ready={len(boulder_ready)}, "
                f"boulder_charging={len(boulder_charging)}, "
                f"pink_target={len(pink_target)}"
            )

            now = time.time()

            # 5. 自動拖曳
            if boulder_ready and pink_target and (now - last_drag_time >= DRAG_COOLDOWN):
                src = boulder_ready[0]
                dst = pink_target[0]

                src_screen = board_to_screen(src["center"], board_roi, window_region)
                dst_screen = board_to_screen(dst["center"], board_roi, window_region)

                print(
                    f"drag: boulder_ready ({src['row']},{src['col']}) "
                    f"-> pink_target ({dst['row']},{dst['col']})"
                )
                print(f"screen: {src_screen} -> {dst_screen}")

                if AUTO_DRAG:
                    before_board = board_img.copy()

                    drag_dice(
                        src_screen,
                        dst_screen,
                        move_duration=0.35,
                        hold_before_drag=0.18
                    )

                    time.sleep(0.4)

                    # 拖曳後再截一次看有沒有變化
                    window_img_after = capture_window(window_region)
                    after_board = crop_board(window_img_after, board_roi)

                    changed = has_board_changed(before_board, after_board)
                    print(f"drag success? {'YES' if changed else 'NO'}")

                    last_drag_time = now

            # 6. 鍵盤控制
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("d"):
                # 手動拖一次
                if boulder_ready and pink_target:
                    src = boulder_ready[0]
                    dst = pink_target[0]

                    src_screen = board_to_screen(src["center"], board_roi, window_region)
                    dst_screen = board_to_screen(dst["center"], board_roi, window_region)

                    print(
                        f"[MANUAL DRAG] boulder_ready ({src['row']},{src['col']}) "
                        f"-> pink_target ({dst['row']},{dst['col']})"
                    )
                    print(f"screen: {src_screen} -> {dst_screen}")

                    before_board = board_img.copy()

                    drag_dice(
                        src_screen,
                        dst_screen,
                        move_duration=0.35,
                        hold_before_drag=0.18
                    )

                    time.sleep(0.4)

                    window_img_after = capture_window(window_region)
                    after_board = crop_board(window_img_after, board_roi)

                    changed = has_board_changed(before_board, after_board)
                    print(f"manual drag success? {'YES' if changed else 'NO'}")

                    last_drag_time = time.time()
                else:
                    print("沒有找到可拖曳的巨石骰或粉色目標骰")

            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("錯誤:", e)
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()