#!/usr/bin/env python3
"""Render a Markdown experiment report as a Korean-capable PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from weasyprint import HTML


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; @bottom-center { content: counter(page); } }
body { font-family: "Noto Sans CJK KR", sans-serif; color: #182230; font-size: 9.5pt; line-height: 1.58; }
h1 { font-size: 20pt; color: #14213d; margin: 0 0 14mm; word-break: keep-all; }
h2 { font-size: 15pt; color: #1d3557; border-bottom: 1px solid #ccd5e0; margin-top: 9mm; }
h3 { font-size: 11.5pt; color: #294c70; margin-top: 6mm; }
code { font-family: "Noto Sans Mono CJK KR", monospace; font-size: 8pt; background: #f3f5f7; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f3f5f7; padding: 3mm; border-radius: 2mm; }
table { width: 100%; border-collapse: collapse; margin: 4mm 0; font-size: 8.5pt; }
th, td { border: 1px solid #cbd3dc; padding: 2mm; vertical-align: top; }
th { background: #eaf0f6; }
blockquote { margin-left: 0; padding-left: 4mm; border-left: 3px solid #6b8db3; color: #46596d; }
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    body = markdown.markdown(
        args.source.read_text(encoding="utf-8"),
        extensions=["fenced_code", "tables", "sane_lists"],
    )
    html = f"<!doctype html><html lang='ko'><meta charset='utf-8'><style>{CSS}</style><body>{body}</body></html>"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(args.source.parent)).write_pdf(str(args.output))


if __name__ == "__main__":
    main()
