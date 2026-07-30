"""Generate the portrait-led SVG banners for Suganth's GitHub profile.

Source inputs stay deliberately separate from the generated SVGs: facecard.jpeg
and the three downloaded Simple Icons references are the editable source of truth.
"""
from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUT = ASSETS / "banner"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1180, 610
PX, PY, PW, PH = 55, 132, 330, 374
DOT_X, DOT_Y, DOT_W, DOT_H = 70, 150, 300, 340
RNG = np.random.default_rng(20260730)


def source_logo(name: str) -> str:
    """Read the actual downloaded Simple Icons path; never redraw a brand mark."""
    text = (ASSETS / f"{name}.svg").read_text(encoding="utf-8")
    return re.search(r'<path d="([^"]+)"', text).group(1)


def dither(mode: str) -> np.ndarray:
    # Head + shoulders crop; its broad framing avoids the aggressive close crop.
    im = Image.open(ROOT / "facecard.jpeg").convert("RGB")
    side = min(im.width, im.height)
    left = (im.width - int(side * 0.88)) // 2
    box = (left, 0, left + int(side * 0.88), side)
    im = im.crop(box).resize((300, 340), Image.Resampling.LANCZOS)
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Contrast(im).enhance(1.3)
    im = im.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    rgb = np.asarray(im, dtype=np.uint8)
    gray = np.asarray(im.convert("L"), dtype=np.float32)

    if mode == "dark":
        # Colour-distance foreground segmentation, closing, hole fill, largest component.
        border = np.concatenate((rgb[:18].reshape(-1, 3), rgb[-18:].reshape(-1, 3),
                                 rgb[:, :18].reshape(-1, 3), rgb[:, -18:].reshape(-1, 3)))
        bg = np.median(border, axis=0)
        distance = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
        yy, xx = np.mgrid[:340, :300]
        prior = ((xx - 151) / 132) ** 2 + ((yy - 180) / 195) ** 2 < 1
        mask = ((distance > 36) & prior).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        flood = mask.copy()
        cv2.floodFill(flood, None, (0, 0), 255)
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(flood))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count > 1:
            keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == keep).astype(np.uint8) * 255
        # The dark version renders the lit subject; clear diffusion at mask edges.
        gray = 255 - gray
        gray[mask == 0] = 255

    # One-bit Floyd-Steinberg dither, with alternate (serpentine) row direction.
    work = gray.copy()
    out = np.zeros_like(work, dtype=bool)
    for y in range(work.shape[0]):
        direction = 1 if y % 2 == 0 else -1
        for x in range(0, work.shape[1]) if direction == 1 else range(work.shape[1] - 1, -1, -1):
            old = work[y, x]
            new = 255 if old >= 128 else 0
            out[y, x] = new == 0
            err = old - new
            for dx, dy, weight in ((direction, 0, 7 / 16), (-direction, 1, 3 / 16),
                                   (0, 1, 5 / 16), (direction, 1, 1 / 16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 300 and ny < 340:
                    work[ny, nx] += err * weight
    return out


def compact_path(points: np.ndarray) -> str:
    # Pixel-sized path runs remain crisp when scaled down, unlike font glyph dots.
    return "".join(f"M{DOT_X + x:.1f},{DOT_Y + y:.1f}h.75v.75h-.75z" for x, y in points)


def distribute(points: np.ndarray, n: int) -> list[np.ndarray]:
    choice = RNG.choice(len(points), size=min(n, len(points)), replace=False)
    selected = points[choice]
    return [selected[i::n] for i in range(n)]


def logo_dot_path(kind: int) -> str:
    """900 travelling dots with a fixed command count so SMIL can morph their paths."""
    t = np.linspace(0, 2 * np.pi, 900, endpoint=False)
    if kind == 0:  # Flutter-like slanted flight
        x = 202 + 78 * ((t / (2 * np.pi)) - .5)
        y = 320 + 76 * np.abs(((t / (2 * np.pi)) * 2 % 1) - .5)
    elif kind == 1:  # cloud silhouette distribution
        x = 202 + 87 * np.cos(t) * (0.55 + 0.45 * np.sin(3 * t) ** 2)
        y = 320 + 49 * np.sin(t) - 10 * np.cos(2 * t)
    else:  # Gemini-like radial point distribution
        r = 13 + 86 * ((np.arange(900) * 47) % 900) / 900
        x = 202 + r * np.cos(t * 4)
        y = 320 + r * np.sin(t * 4)
    return "".join(f"M{a:.1f},{b:.1f}h1v1h-1z" for a, b in zip(x, y))


def row(label: str, value: str, y: int) -> str:
    label_escaped, value_escaped = html.escape(label.upper()), html.escape(value)
    # Lock the measured value rather than stretching short values across the panel.
    width = max(128, min(470, len(value) * 8.35))
    return (f'<text x="455" y="{y}" class="label">{label_escaped}</text>'
            f'<line x1="{555 + min(80, len(label) * 3)}" y1="{y - 4}" x2="{1045 - min(220, len(value) * 6)}" y2="{y - 4}" class="leader"/>'
            f'<text x="1045" y="{y}" class="value" text-anchor="end" textLength="{width}" lengthAdjust="spacingAndGlyphs">{value_escaped}</text>')


def render(mode: str) -> str:
    dots = np.argwhere(dither(mode))[:, ::-1]
    # Preserve the 300×340 dither source but cap SVG ink at ~17k dots.
    if len(dots) > 17000:
        dots = dots[RNG.choice(len(dots), 17000, replace=False)]
    np.save(OUT / f"portrait-{mode}.npy", dots)
    intro = distribute(dots, 60)
    loop = distribute(dots, 94)
    intro_svg = "".join(
        f'<path d="{compact_path(group)}" class="portrait" opacity="0"><animate attributeName="opacity" begin="{.18 + i * .032:.3f}s" dur="1.25s" fill="freeze" values="0;1"/></path>'
        for i, group in enumerate(intro))
    loop_svg = ""
    for i, group in enumerate(loop):
        noise = RNG.normal(0, 4, 2)
        dx, dy = (202 - np.mean(group[:, 0])) * .42 + noise[0], (320 - np.mean(group[:, 1])) * .42 + noise[1]
        loop_svg += (f'<g opacity="0"><path d="{compact_path(group)}" class="portrait"/>'
                     f'<animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="0;1;1;0;1" keyTimes="0;.001;.211;.303;1"/>'
                     f'<animateTransform attributeName="transform" type="translate" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="0 0;0 0;{dx:.1f} {dy:.1f};0 0;0 0" keyTimes="0;.211;.303;.394;1"/></g>')
    logos = [("flutter", "FLUTTER"), ("googlecloud", "GOOGLE CLOUD"), ("googlegemini", "GEMINI")]
    logo_svg = ""
    for i, (slug, label) in enumerate(logos):
        start = [4.5, 7.8, 11.1][i]
        logo_svg += (f'<g transform="translate(112 230) scale(7.5)" opacity="0" class="logo">'
                     f'<path d="{source_logo(slug)}"/>'
                     f'<animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="0;{(start-3.2)/14.2:.3f};{(start-3.2+.35)/14.2:.3f};{(start-3.2+2.0)/14.2:.3f};{(start-3.2+2.45)/14.2:.3f};1"/></g>'
                     f'<text x="220" y="455" class="logo-name" text-anchor="middle" opacity="0">{label}<animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="0;0;1;1;0;0" keyTimes="0;{(start-3.2)/14.2:.3f};{(start-3.2+.35)/14.2:.3f};{(start-3.2+2.0)/14.2:.3f};{(start-3.2+2.45)/14.2:.3f};1"/></text>')
    travellers = (f'<path d="{logo_dot_path(0)}" class="traveller" opacity="0">'
                  f'<animate attributeName="d" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="{logo_dot_path(0)};{logo_dot_path(0)};{logo_dot_path(1)};{logo_dot_path(1)};{logo_dot_path(2)};{logo_dot_path(2)};{logo_dot_path(0)}" keyTimes="0;.211;.303;.444;.535;.676;1"/>'
                  '<animate attributeName="opacity" begin="3.2s" dur="14.2s" repeatCount="indefinite" values="0;0;1;1;1;1;0" keyTimes="0;.211;.303;.444;.535;.676;1"/></path>')
    rows = "".join([
        row("SUBJECT", "SUGANTH K", 182),
        row("ROLE", "AI Engineer · Full-Stack Developer", 205),
        row("ORIGIN", "Coimbatore, India", 228),
        row("EDUCATION", "B.Tech CSE (Artificial Intelligence), 2023–2027", 251),
        row("STATUS", "Building + Learning + Shipping", 286),
        row("TOOLCHAIN", "VS Code · Git · Android Studio · Google Cloud", 309),
        row("CORE.LANG", "Python · JavaScript · Dart · Java · C++ · SQL", 344),
        row("CORE.FRONTEND", "Flutter · React · React Native · Next.js", 367),
        row("CORE.BACKEND", "Node.js · Express · Firebase · LangChain", 390),
        row("CORE.DATABASE", "MongoDB · SQLite · Firebase", 413),
        row("CORE.INFRA", "Google Cloud · Linux · GitHub Actions", 436),
        row("GRID.MAIL", "suganthk2005@gmail.com", 471),
        row("GRID.PORTFOLIO", "coming soon", 494),
    ])
    bg, panel, ink, muted, chrome = ("#0A101F", "#0E1930", "#A78BFA", "#7C8AA5", "#22D3EE") if mode == "dark" else ("#F7F4FF", "#FFFFFF", "#5B21B6", "#536079", "#0891B2")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" viewBox="0 0 1180 610" role="img" aria-label="Animated system profile for Suganth K">
<style>
.bg{{fill:{bg}}}.window{{fill:{panel};stroke:{chrome};stroke-width:1.5}}.chrome{{fill:{chrome}}}.portrait,.traveller,.logo{{fill:{ink}}}.traveller{{fill:#10B981}}.label{{font:12px monospace;fill:{muted};letter-spacing:1px}}.value{{font:14px monospace;fill:{ink}}}.leader{{stroke:{muted};stroke-dasharray:1 5;opacity:.8}}.title{{font:13px monospace;fill:{ink};letter-spacing:1px}}.logo-name{{font:12px monospace;fill:{chrome};letter-spacing:2px}}</style>
<rect width="1180" height="610" class="bg"/><rect x="18" y="18" width="1144" height="574" rx="14" class="window"/>
<path d="M18 65H1162" stroke="{chrome}" opacity=".55"/><circle cx="43" cy="42" r="6" fill="#EF4444"><animate attributeName="opacity" values="1;.25;1" dur="1.2s" repeatCount="indefinite"/></circle><circle cx="63" cy="42" r="6" fill="#F59E0B"/><circle cx="83" cy="42" r="6" fill="#10B981"/>
<text x="112" y="47" class="title">profile.sh --live</text><rect x="1012" y="31" width="112" height="23" rx="11" fill="#EF4444"/><text x="1068" y="47" fill="white" text-anchor="middle" style="font:12px monospace;letter-spacing:1px">● LIVE</text>
<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" rx="8" fill="none" stroke="{chrome}" opacity=".7"/><text x="{PX}" y="112" class="title">VISUAL.MAP</text><text x="220" y="112" class="title" text-anchor="middle">01 / 01</text>
<g shape-rendering="crispEdges">{intro_svg}</g><g shape-rendering="crispEdges">{loop_svg}</g>{travellers}{logo_svg}
<text x="455" y="112" class="title">SYSTEM.INFO</text><rect x="930" y="94" width="115" height="25" rx="12" fill="#10B981"/><text x="987" y="112" text-anchor="middle" fill="{bg}" style="font:14px monospace;font-weight:bold">@suganth07</text>{rows}
</svg>'''


if __name__ == "__main__":
    for variant in ("dark", "light"):
        (OUT / f"{variant}.svg").write_text(render(variant), encoding="utf-8")
    print(f"Wrote {OUT / 'dark.svg'} and {OUT / 'light.svg'}")
