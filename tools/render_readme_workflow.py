#!/usr/bin/env python3
"""Build the self-contained README workflow illustrations (no dependencies)."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
PALETTES = {
    "light": dict(bg="#f4f0e5", panel="#faf7ef", ink="#14212b", muted="#625e53",
                  gold="#7e5c22", line="#d1be92", ornament="#a2854e"),
    "dark": dict(bg="#111923", panel="#18222e", ink="#f3eddf", muted="#c2bba9",
                 gold="#dfbd7a", line="#5d543e", ornament="#a78d59"),
}
COPY = {
    "en": {
        "eyebrow": "MEASUREMENT WORKFLOW",
        "title": "Raw measurements to scored predictions",
        "desc": "Prepare device features and training targets, inspect the plan, fit and predict, then score later outcomes without refitting.",
        "steps": [
            ("Prepare", "Join exact device IDs", "Use training targets only"),
            ("Plan", "Check declared groups", "Preview the fit count"),
            ("Run", "Fit on training data", "Save held-out predictions"),
            ("Score", "Match IDs to outcomes", "Score without refitting"),
        ],
        "boundary": "LATER OUTCOMES",
        "note": "Join later labels only at Score. No new fits.",
        "mobile_note": ["Join later labels only at Score.", "No new fits."],
    },
    "zh-TW": {
        "eyebrow": "量測工作流程",
        "title": "從原始量測到預測評分",
        "desc": "整理元件特徵與訓練端點、檢查計畫、擬合並預測，最後以日後觀測值計分，不重新擬合。",
        "steps": [
            ("整理", "按元件 ID 精確配對", "只用訓練端點"),
            ("規劃", "檢查已宣告的群組", "預覽擬合次數"),
            ("執行", "以訓練資料擬合", "保存留出組的預測"),
            ("評分", "依 ID 對照後續觀測", "只評分，不重擬合"),
        ],
        "boundary": "後續觀測值",
        "note": "只在評分時合入後續觀測值，不重擬合。",
        "mobile_note": ["只在評分時合入後續觀測值。", "不重擬合。"],
    },
}
COMMANDS = ("prepare", "plan", "run", "score")


def text(x, y, value, *, size=22, fill="ink", family="sans", anchor="start", spacing=0):
    families = {
        "sans": "system-ui, -apple-system, 'Segoe UI', sans-serif",
        "serif": "Georgia, 'Times New Roman', serif",
        "mono": "ui-monospace, 'SFMono-Regular', Consolas, monospace",
    }
    return (f'<text x="{x}" y="{y}" fill="{{{fill}}}" font-size="{size}" '
            f'font-family="{families[family]}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing}">{escape(value)}</text>')


def icon(index, x, y):
    """Original small line ornaments; no measurements or scientific claims."""
    drawings = [
        '<path d="M-20 12V-12M0 18V-18M20 12V-12"/>'
        '<circle cx="-20" cy="0" r="4"/><circle cx="0" cy="-8" r="4"/>'
        '<circle cx="20" cy="6" r="4"/>',
        '<path d="M-20-16H20V16H-20ZM-7-16V16M7-16V16M-20-5H20M-20 5H20"/>',
        '<path d="M0-23V23M-15-15C24-15-24 15 15 15M15-15C-24-15 24 15-15 15"/>'
        '<path d="M-6-23H6M-6 23H6"/>',
        '<path d="M-20 18V-18H7L20-5V18ZM7-18V-5H20M-11 5L-3 12L11-1"/>',
    ]
    return (f'<g transform="translate({x} {y})" fill="none" stroke="{{gold}}" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{drawings[index]}</g>')


def render(language, theme, mobile):
    copy = COPY[language]
    w, h = (480, 906) if mobile else (1200, 454)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc" '
           f'xml:lang="{language}">',
           f'<title id="title">{copy["title"]}</title>',
           f'<desc id="desc">{copy["desc"]}</desc>',
           f'<rect width="{w}" height="{h}" rx="12" fill="{{bg}}"/>',
           f'<path d="M24 1H{w-24}M24 {h-1}H{w-24}" stroke="{{line}}"/>']
    if mobile:
        out += [text(24, 38, copy["eyebrow"], size=15, fill="gold", spacing=2),
                text(24, 76, copy["title"], size=25, family="serif")]
        for i, (title, first, second) in enumerate(copy["steps"]):
            y = 104 + 160 * i
            if i < 3:
                out.append(f'<path d="M60 {y+138}V{y+160}" stroke="{{ornament}}"/>')
            out += [f'<rect x="24" y="{y}" width="432" height="140" rx="8" '
                    'fill="{panel}" stroke="{line}"/>',
                    f'<circle cx="60" cy="{y+39}" r="20" fill="{{bg}}" stroke="{{line}}"/>',
                    text(60, y+45, f"0{i+1}", size=16, family="mono", fill="gold", anchor="middle"),
                    text(100, y+43, title, size=28, family="serif"),
                    icon(i, 418, y+39),
                    text(100, y+82, first, size=21),
                    text(100, y+114, second, size=21, fill="muted")]
            if language == "zh-TW":
                out.append(text(171, y+41, COMMANDS[i], size=16, family="mono", fill="gold"))
        out += ['<path d="M423 801H446V731" '
                'fill="none" stroke="{gold}" stroke-width="2" stroke-dasharray="5 4"/>',
                '<path d="M439 739L446 731L453 739" fill="none" stroke="{gold}" stroke-width="2"/>',
                '<rect x="24" y="767" width="399" height="116" rx="8" '
                'fill="{panel}" stroke="{line}"/>',
                text(40, 799, copy["boundary"], size=18, family="mono", fill="gold"),
                text(40, 833, copy["mobile_note"][0], size=22, fill="muted"),
                text(40, 862, copy["mobile_note"][1], size=22, fill="muted")]
    else:
        out += [text(32, 42, copy["eyebrow"], size=16, fill="gold", spacing=3),
                text(32, 84, copy["title"], size=30, family="serif"),
                '<path d="M1050 43H1158M1104 30V56" stroke="{ornament}"/>',
                '<path d="M1104 34L1113 43L1104 52L1095 43Z" fill="{bg}" stroke="{gold}"/>']
        for i, (title, first, second) in enumerate(copy["steps"]):
            x = 32 + 290 * i
            out += [f'<rect x="{x}" y="116" width="266" height="214" rx="8" '
                    'fill="{panel}" stroke="{line}"/>',
                    text(x+20, 152, f"0{i+1}", size=16, family="mono", fill="gold"),
                    icon(i, x+223, 153),
                    text(x+20, 211, title, size=32, family="serif"),
                    text(x+20, 260, first, size=20),
                    text(x+20, 294, second, size=20, fill="muted")]
            if language == "zh-TW":
                out.append(text(x+99, 209, COMMANDS[i], size=17, family="mono", fill="gold"))
            if i < 3:
                out.append(f'<path d="M{x+267} 223H{x+287}M{x+282} 218L{x+287} 223L{x+282} 228" '
                           'fill="none" stroke="{ornament}" stroke-linecap="round"/>')
        out += ['<path d="M1035 367V331M1028 339L1035 331L1042 339" '
                'fill="none" stroke="{gold}" stroke-width="2" stroke-dasharray="5 4"/>',
                '<rect x="32" y="367" width="1136" height="62" rx="8" '
                'fill="{panel}" stroke="{line}"/>',
                text(52, 405, copy["boundary"], size=18, family="mono", fill="gold"),
                text(375, 405, copy["note"], size=21, fill="muted")]
    out.append('</svg>\n')
    return "\n".join(out).format(**PALETTES[theme])


def assets():
    for language in COPY:
        for theme in PALETTES:
            for mobile in (False, True):
                suffix = "-mobile" if mobile else ""
                yield (ASSETS / f"clotho-workflow-{language}-{theme}{suffix}.svg",
                       render(language, theme, mobile))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without writing any files.")
    args = parser.parse_args()
    stale = []
    for path, content in assets():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(content, encoding="utf-8")
    if stale:
        parser.exit(1, "Stale workflow assets:\n" + "\n".join(stale) + "\n")
    print("8 workflow assets checked." if args.check else "8 workflow assets rendered.")


if __name__ == "__main__":
    main()
