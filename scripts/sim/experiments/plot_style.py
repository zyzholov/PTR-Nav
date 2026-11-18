"""Shared Chinese labels, font validation and binomial confidence intervals."""

import math
from pathlib import Path

import matplotlib
from matplotlib import font_manager


SCENARIO_NAMES = {
    "flat": "平地", "ramp": "缓坡", "directional_slope": "方向性斜坡",
    "steep_slope": "陡坡", "step": "不可通过台阶", "ridge": "窄脊",
    "rough": "粗糙地形", "small_obstacle": "小型凸起",
    "detour": "静态绕障", "crossing": "动态横穿", "emerge": "突然出现障碍",
    "clearance": "动态残影清除", "clearance_guard": "残影清除与静态保护",
    "mars": "火星地形", "unknown": "未分类",
}


def scenario_name(value):
    return SCENARIO_NAMES.get(value, value)


def configure_chinese():
    candidates = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                  Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font = font_manager.FontProperties(fname=str(path))
            name = font.get_name()
            break
    else:
        name = next((f.name for f in font_manager.fontManager.ttflist
                     if any(s in f.name for s in ("CJK", "SimHei", "WenQuanYi"))), None)
        if name is None:
            raise RuntimeError("找不到中文字体，请安装 fonts-noto-cjk 后再绘图")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [name, "DejaVu Sans"],
        "axes.unicode_minus": False, "pdf.fonttype": 42,
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
    })
    return name


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return None, None
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == total else min(1.0, center + half)
    return lower, upper
