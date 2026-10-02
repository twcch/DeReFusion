# -*- coding: utf-8 -*-
"""
汇总生成器：把 result_long_term_forecast.txt + 所有 volatility_stratification_*.json
整理成一份可读摘要（afternoon-digest.md），供 18:00 单次唤醒时直接阅读。

零 LLM 成本：纯文件后处理。
"""
import json
import os
import re
from glob import glob

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
OUT = os.path.join(REPO, "reproduction", "results", "afternoon-digest.md")

SETTING_RE = re.compile(
    r"long_term_forecast_(?P<asset>[A-Z]+)_96_(?P<pl>\d+)_(?P<model>[A-Za-z0-9\-]+)_custom_ftMS_.*_seed(?P<seed>\d+)_0"
)


def parse_metrics_log(path):
    """解析一行式结果日志 -> [ {asset, pl, model, seed, metrics{...}} ]"""
    rows = []
    if not os.path.exists(path):
        return rows
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="ignore") if l.strip()]
    for i, line in enumerate(lines):
        m = SETTING_RE.search(line)
        if not m or i + 1 >= len(lines):
            continue
        nxt = lines[i + 1]
        if not nxt.startswith("mse:"):
            continue
        kv = {}
        for part in nxt.rstrip(",").split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                try:
                    kv[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        rows.append({"asset": m.group("asset"), "pl": int(m.group("pl")),
                     "model": m.group("model"), "seed": int(m.group("seed")), "metrics": kv})
    return rows


def _inter(comp):
    """兼容两代键名：优先 50/50 口径，其次任务书 Q4+Q5 口径，最后旧键。"""
    for k in ("interaction_high50_minus_low50", "interaction_Q45_minus_Q12", "interaction"):
        v = comp.get(k)
        if v:
            return v
    return None


def verdict_for(strata, comp):
    """机械判定：高波动分位非线性显著占优 & 平静分位不显著 -> strong/full support"""
    def sig_neg(name):
        c = comp.get(name)
        return bool(c) and c["ci95"][1] < 0

    def ns(name):
        c = comp.get(name)
        return bool(c) and c["ci95"][0] < 0 < c["ci95"][1]

    hi = sig_neg("high20%") or sig_neg("high50%")
    lo = ns("low20%") or ns("low50%")
    inter = _inter(comp)
    inter_sig = bool(inter) and (inter["ci95"][1] < 0 or inter["ci95"][0] > 0)
    if hi and lo and inter_sig:
        return "A（支持：高波动显著、平静打平、交互显著）"
    if hi and lo:
        return "A-（支持但交互不显著）"
    if hi:
        return "C（高波动占优，平静未打平 → 需更多资产）"
    return "B（未复现：高波动分位无显著占优）"


def main():
    out = ["# 午后批次摘要（自动生成）", ""]

    rows = parse_metrics_log(os.path.join(REPO, "result_long_term_forecast.txt"))
    if rows:
        out.append("## 1. 总览指标（归一化尺度，全部运行）")
        out.append("")
        out.append("| 资产 | T | 模型 | seed | MSE | MAE | RMSE | MAPE | MSPE | R² |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            m = r["metrics"]
            out.append(f"| {r['asset']} | {r['pl']} | {r['model']} | {r['seed']} | "
                       f"{m.get('mse', float('nan')):.5f} | {m.get('mae', float('nan')):.5f} | "
                       f"{m.get('rmse', float('nan')):.5f} | {m.get('mape', float('nan')):.5f} | "
                       f"{m.get('mspe', float('nan')):.5f} | {m.get('r2', float('nan')):.4f} |")
        out.append("")

    out.append("## 2. 工况分层（按相对波动率为主判定口径）")
    out.append("")
    files = sorted(glob(os.path.join(REPO, "reproduction", "results", "volatility_stratification_*.json")))
    for f in files:
        try:
            R = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        asset = R.get("asset")
        for hkey, H in R.get("horizons", {}).items():
            strata, comp = H["strata"], H["comparison"]
            out.append(f"### {asset} {hkey} — {os.path.basename(f)}")
            out.append("")
            out.append("| 分层 | n | DeReFusion MSE | DLinear MSE | 相对改善 | ΔMSE 95%CI | 胜率 |")
            out.append("|---|---|---|---|---|---|---|")
            for name in ("low20%", "low50%", "high50%", "high20%"):
                if name not in strata:
                    continue
                c = comp.get(name, {})
                rel = (strata[name]["revin-DLinear"]["MSE"] - strata[name]["DeReFusion"]["MSE"]) \
                    / strata[name]["revin-DLinear"]["MSE"] * 100
                ci = c.get("ci95", [float("nan"), float("nan")])
                out.append(f"| {name} | {c.get('n','')} | {strata[name]['DeReFusion']['MSE']:.5f} | "
                           f"{strata[name]['revin-DLinear']['MSE']:.5f} | {rel:+.1f}% | "
                           f"[{ci[0]:+.5f},{ci[1]:+.5f}] | {c.get('win_rate',0):.1%} |")
            it = _inter(comp)
            if it:
                out.append("")
                out.append(f"- 交互效应 Δ_high50−Δ_low50 = {it['value']:+.5f}, 95%CI[{it['ci95'][0]:+.5f},{it['ci95'][1]:+.5f}]")
            out.append(f"- **判定：{verdict_for(strata, comp)}**")
            out.append("")

    txt = "\n".join(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] {OUT} ({len(txt)} chars)")


if __name__ == "__main__":
    main()
