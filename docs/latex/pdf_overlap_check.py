# -*- coding: utf-8 -*-
"""PDF overlap QA: find text that overlaps across columns or spills past the margins.

Method: take every word's bounding box; flag any pair whose x-ranges overlap by more than a small
tolerance AND whose y-ranges overlap (i.e. glyphs occupying the same place), and flag words whose box
crosses the column gutter. Works for two-column IEEEtran output.

Usage: python pdf_overlap_check.py <pdf> [<pdf> ...]
"""
from __future__ import annotations

import sys

import pdfplumber

TOL_X = 1.0     # points of horizontal overlap tolerated
TOL_Y = 2.0


def check(path):
    print(f"=== {path}")
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            words = [w for w in page.extract_words(use_text_flow=False) if w["text"].strip()]
            mid = page.width / 2
            # column gutter edges (IEEEtran two-column): approximate from the page geometry
            left_col_right = max((w["x1"] for w in words if w["x1"] <= mid + 30), default=0)
            right_col_left = min((w["x0"] for w in words if w["x0"] > mid), default=page.width)

            # 1) cross-column overlap: a box that starts left of the gutter and ends past it
            spills = [w for w in words if w["x0"] < mid - 5 and w["x1"] > mid + 5]
            # 2) same-place glyph collisions, bucketed by y for speed
            buckets = {}
            for w in words:
                buckets.setdefault(round(w["top"] / 6), []).append(w)
            collisions = []
            for b, ws in buckets.items():
                for dy in (0, 1):
                    for w in ws:
                        for v in buckets.get(b + dy, []):
                            if w is v:
                                continue
                            if (min(w["x1"], v["x1"]) - max(w["x0"], v["x0"]) > TOL_X
                                    and min(w["bottom"], v["bottom"]) - max(w["top"], v["top"]) > TOL_Y):
                                collisions.append((w["text"], v["text"], round(w["x0"]), round(w["top"])))
            print(f"  page {pno}: words={len(words)} gutter≈[{left_col_right:.0f},{right_col_left:.0f}] "
                  f"spills={len(spills)} collisions={len(collisions)}")
            for w in spills[:6]:
                print(f"     SPILL  x0={w['x0']:.0f} x1={w['x1']:.0f} top={w['top']:.0f} "
                      f"{w['text'][:40]!r}")
            for a, b, x, y in collisions[:6]:
                print(f"     COLLIDE {a[:28]!r} <> {b[:28]!r} at x={x} y={y}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
