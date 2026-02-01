#!/usr/bin/env python3
"""
Generate a single Markdown report that summarizes the categorized (codebook-only) results
and embeds the saved figures.

Assumes you already ran:
  1) scripts/category_stats_from_token_level.py  -> writes analysis/categorized/*.csv
  2) scripts/plot_category_outputs.py            -> writes analysis/categorized/figures/*.png

Output:
  analysis/categorized/report.md
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def md_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    return df.head(max_rows).to_markdown(index=False) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis/categorized")
    ap.add_argument("--figdir", default="analysis/categorized/figures")
    ap.add_argument("--out", default="analysis/categorized/report.md")
    ap.add_argument("--cluster_col", default="group", help="group or long_cluster (auto-fallback if missing)")
    ap.add_argument("--topk", type=int, default=15, help="Top lemmas to show per category and POS")
    args = ap.parse_args()

    indir = Path(args.indir)
    figdir = Path(args.figdir)
    out = Path(args.out)

    # Load CSVs
    cat_tot = pd.read_csv(indir / "category_totals.csv")
    cat_sec = pd.read_csv(indir / "category_by_section.csv")
    cat_clu = pd.read_csv(indir / "category_by_cluster.csv")
    pos_cat = pd.read_csv(indir / "pos_by_category.csv")
    top_lem = pd.read_csv(indir / "top_lemmas_by_category.csv")

    # cluster col detection
    cluster_col = args.cluster_col
    if cluster_col not in cat_clu.columns:
        if "long_cluster" in cat_clu.columns:
            cluster_col = "long_cluster"
        else:
            cluster_col = [c for c in cat_clu.columns if c not in {"category","token_count","cluster_total","share_in_cluster"}][0]

    # Agency index (on shares)
    piv = cat_clu.pivot_table(index=cluster_col, columns="category", values="share_in_cluster", aggfunc="sum", fill_value=0)
    num = piv.get("CONTROL", 0) + piv.get("AUTHORSHIP", 0)
    den = piv.get("AI_USE", 0) + piv.get("IDEATION", 0)
    agency = (num / den.replace(0, pd.NA)).fillna(0).rename("agency_index").reset_index()
    agency = agency.sort_values("agency_index", ascending=False)

    # Top lemmas per category & POS (from top_lemmas_by_category.csv)
    top_lem = top_lem.sort_values(["category", "pos_group_cb", "token_count"], ascending=[True, True, False])

    categories = list(top_lem["category"].dropna().unique())
    pos_vals = list(top_lem["pos_group_cb"].dropna().unique())

    # Build report
    lines = []
    lines.append("# Categorized Lexis Report (codebook-only)\n")
    lines.append("This report summarizes counts and shares **restricted to the annotated codebook lemmas only**.\n")
    lines.append("Categories: **AI_USE, CONTROL, AUTHORSHIP, IDEATION**.\n")

    lines.append("## Overview\n")
    lines.append("### Category totals (token counts)\n")
    lines.append(md_table(cat_tot, max_rows=50))

    lines.append("### Agency Index by cluster\n")
    lines.append("Agency Index = (CONTROL + AUTHORSHIP) / (AI_USE + IDEATION)\n")
    lines.append(md_table(agency, max_rows=50))

    lines.append("## Figures\n")
    fig_map = [
        ("Category share by section", "category_share_by_section.png"),
        ("Category share by cluster", "category_share_by_cluster.png"),
        ("Category share within VERB", "category_share_within_VERB.png"),
        ("Category share within NOUN", "category_share_within_NOUN.png"),
        ("Agency Index by cluster", "agency_index_by_cluster.png"),
    ]
    for title, fname in fig_map:
        fpath = figdir / fname
        if fpath.exists():
            lines.append(f"### {title}\n")
            # relative path so it renders nicely in repo
            rel = f"{figdir.name}/{fname}" if figdir.name == "figures" else str(fpath)
            lines.append(f"![{title}]({rel})\n")
        else:
            lines.append(f"### {title}\n")
            lines.append(f"_(missing figure: {fname})_\n")

    lines.append("## Top lemmas per category\n")
    for cat in categories:
        lines.append(f"### {cat}\n")
        for pos in pos_vals:
            sub = top_lem[(top_lem["category"] == cat) & (top_lem["pos_group_cb"] == pos)].copy()
            if sub.empty:
                continue
            sub = sub.rename(columns={"token_count": "count"})
            lines.append(f"**{pos} (top {args.topk})**\n")
            lines.append(md_table(sub[["lemma", "count"]], max_rows=args.topk))

    lines.append("## Data files\n")
    lines.append("- category_by_section.csv\n")
    lines.append("- category_by_cluster.csv\n")
    lines.append("- pos_by_category.csv\n")
    lines.append("- top_lemmas_by_category.csv\n")
    lines.append("- top_lemmas_by_category_by_cluster.csv\n")
    lines.append("- top_lemmas_by_category_by_section.csv\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
