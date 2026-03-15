import pygetwindow as gw


def list_visible_windows():
    windows = []
    for w in gw.getAllWindows():
        title = w.title.strip()
        if title:
            windows.append(w)
    return windows


def choose_window():
    windows = list_visible_windows()

    if not windows:
        raise RuntimeError("找不到任何可用視窗")

    print("=== 目前視窗列表 ===")
    for i, w in enumerate(windows):
        print(f"[{i}] {w.title}")

    idx = int(input("請輸入模擬器視窗編號: "))
    win = windows[idx]

    try:
        win.activate()
    except Exception:
        pass

    return win


def get_window_region(win):
    if win.width <= 0 or win.height <= 0:
        raise RuntimeError("視窗大小異常，請確認視窗不是最小化")

    return {
        "left": win.left,
        "top": win.top,
        "width": win.width,
        "height": win.height,
    }