import win32gui
import win32con


EMULATOR_KEYWORDS = [
    "BlueStacks",
    "BlueStacks App Player",
    "LDPlayer",
    "MuMu",
    "Nox",
    "MEmu",
    "GameLoop",
    "雷電",
    "夜神",
    "模擬器",
    "dnplayer",
]


def _is_candidate_window(hwnd):
    if not win32gui.IsWindowVisible(hwnd):
        return False

    if win32gui.IsIconic(hwnd):
        return False

    title = win32gui.GetWindowText(hwnd).strip()
    if not title:
        return False

    # 這邊只做粗略過濾，不在這裡依賴尺寸
    title_lower = title.lower()
    if not any(keyword.lower() in title_lower for keyword in EMULATOR_KEYWORDS):
        return False

    return True


def list_emulator_windows():
    results = []

    def callback(hwnd, _):
        if _is_candidate_window(hwnd):
            title = win32gui.GetWindowText(hwnd).strip()
            results.append({
                "hwnd": hwnd,
                "title": title,
            })

    win32gui.EnumWindows(callback, None)
    return results


def activate_window(hwnd):
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass


def choose_window():
    windows = list_emulator_windows()

    if not windows:
        raise RuntimeError("找不到模擬器視窗")

    print("=== 模擬器視窗列表 ===")
    for i, w in enumerate(windows):
        print(f"[{i}] hwnd={w['hwnd']} | {w['title']}")

    while True:
        try:
            idx = int(input("請輸入要使用的視窗編號: "))
            if 0 <= idx < len(windows):
                chosen = windows[idx]
                activate_window(chosen["hwnd"])
                return chosen
            print("編號超出範圍")
        except ValueError:
            print("請輸入數字")