import pyautogui


pyautogui.FAILSAFE = True


def drag_dice(src_xy, dst_xy, move_duration=0.15):
    pyautogui.moveTo(src_xy[0], src_xy[1], duration=0.05)
    pyautogui.mouseDown()
    pyautogui.moveTo(dst_xy[0], dst_xy[1], duration=move_duration)
    pyautogui.mouseUp()