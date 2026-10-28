"""Record docs/img/demo.gif against the live Space (or a local server): pick a dev shopper → the five lists side by side →
click a "seen" item → its neighbours in the Item tab → the Evidence tab. Screenshots assembled with Pillow; ≤ 30 s, ≤ 3 MB.

    python scripts/demo_gif.py [--base https://davutkursun-shop-recsys.hf.space] [--shopper 664] [--out docs/img/demo.gif] [--frames-dir DIR]
Needs: pip install playwright pillow && python -m playwright install chromium (a throwaway venv is fine)
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 3 * 1024 * 1024
MAX_MS = 30_000


def record(base: str, width: int, height: int, shopper: int) -> list[tuple[Image.Image, int]]:
    from playwright.sync_api import sync_playwright

    frames: list[tuple[Image.Image, int]] = []

    def shot(page, hold_ms: int) -> None:
        frames.append((Image.open(io.BytesIO(page.screenshot(full_page=False))).convert("RGB"), hold_ms))

    def scroll_to(page, selector: str) -> None:
        page.locator(selector).first.scroll_into_view_if_needed()
        time.sleep(0.6)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(base.rstrip("/") + "/ui/", wait_until="networkidle", timeout=180_000)
        page.wait_for_selector("text=What they opened next", timeout=120_000)  # the opening shopper's lists are in
        time.sleep(1.0)
        shot(page, 2000)
        label = f"{shopper:04d} ·"
        box = page.get_by_label("Dev shopper")
        box.click()
        box.press("ControlOrMeta+a")
        box.press_sequentially(f"{shopper:04d}", delay=90)  # 2,000 shoppers: the dropdown filters on keystrokes (fill() does not)
        page.get_by_role("option").filter(has_text=label).first.click(timeout=15_000)
        page.wait_for_selector(f"text=Shopper {shopper:04d}", timeout=60_000)
        time.sleep(1.2)
        shot(page, 2600)  # the shopper, what they opened next, the history the models read
        scroll_to(page, "#list-reranked")
        page.mouse.wheel(0, 120)
        time.sleep(0.6)
        shot(page, 3400)  # the five lists side by side and the "why" line
        page.locator("#list-reranked [role=gridcell]").filter(has_text="seen").first.click()  # Gradio 6 table cells
        page.wait_for_selector("text=Nearest items by cosine", timeout=60_000)
        time.sleep(1.2)
        shot(page, 3000)  # the clicked item's neighbours
        page.get_by_role("tab", name="Evidence").click()
        page.wait_for_selector("text=The test window", timeout=60_000)
        time.sleep(1.5)
        shot(page, 2800)  # the test table with CIs
        page.mouse.wheel(0, 650)
        time.sleep(0.9)
        shot(page, 2600)  # paired differences and the by-history chart
        page.mouse.wheel(0, 900)
        time.sleep(0.9)
        shot(page, 2400)  # breakdowns by subset
        browser.close()
    return frames


def write_gif(frames: list[tuple[Image.Image, int]], out: Path, start_scale: float) -> int:
    scale = start_scale
    while True:
        imgs = [f.resize((int(f.width * scale), int(f.height * scale)), Image.LANCZOS).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f, _ in frames]
        buf = io.BytesIO()
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=[h for _, h in frames], loop=0, optimize=True)
        if buf.tell() <= MAX_BYTES or scale < 0.45:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(buf.getvalue())
            return buf.tell()
        scale *= 0.9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://davutkursun-shop-recsys.hf.space")
    ap.add_argument("--shopper", type=int, default=664)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "demo.gif")
    ap.add_argument("--frames-dir", type=Path, default=None, help="also save every frame as PNG (for checking)")
    ap.add_argument("--width", type=int, default=1360)
    ap.add_argument("--height", type=int, default=820)
    ap.add_argument("--scale", type=float, default=0.8)
    a = ap.parse_args(argv)
    frames = record(a.base, a.width, a.height, a.shopper)
    if a.frames_dir:
        a.frames_dir.mkdir(parents=True, exist_ok=True)
        for i, (f, _) in enumerate(frames):
            f.save(a.frames_dir / f"frame_{i:02d}.png")
    total_ms = sum(h for _, h in frames)
    size = write_gif(frames, a.out, a.scale)
    print(f"{a.out}: {len(frames)} frames, {total_ms / 1000:.1f} s, {size / 1024:.0f} KB")
    return 0 if total_ms <= MAX_MS and size <= MAX_BYTES else 1


if __name__ == "__main__":
    raise SystemExit(main())
