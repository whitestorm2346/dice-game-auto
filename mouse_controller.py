import time
import pyautogui

pyautogui.FAILSAFE = True


def drag_dice(src_xy, dst_xy, move_duration=0.35, hold_before_drag=0.18):
    pyautogui.moveTo(src_xy[0], src_xy[1], duration=0.15)
    pyautogui.mouseDown()
    time.sleep(hold_before_drag)
    pyautogui.moveTo(dst_xy[0], dst_xy[1], duration=move_duration)
    time.sleep(0.08)
    pyautogui.mouseUp()