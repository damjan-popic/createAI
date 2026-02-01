#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def stacked_share_plot(df: pd.DataFrame, index_col: str, cat_col: str, share_col: str, title: str, outpath: Path):
    pivot = df.pivot_table(index=index_col, columns=cat_col, values=share_col, aggfunc="sum", fill_value=0)
    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.ylabel("Share")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()


def grouped_bar(df: pd.DataFrame, index_col: str, cols: list[str], title: str, ylabel: str, outpath: Path):
    pivot = df.pivot_table(index=index_col, values=cols, aggfunc="sum", fill_value=0)
    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis/categorized")
    ap.add_argument("--figdir", default="analysis/categorized/figures")
    ap.add_argument("--cluster_col", default="group", help="Use 'group' or 'long_cluster' depending on your clustering file")
    args = ap.parse_args()

    indir = Path(args.indir)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # --- 1) Category by section (share)
    cat_sec = pd.read_csv(indir / "category_by_section.csv")
    stacked_share_plot(
        cat_sec, index_col="section", cat_col="category", share_col="share_in_section",
        title="Category share by section (codebook-only)",
        outpath=figdir / "category_share_by_section.png"
    )

    # --- 2) Category by cluster (share)
    cat_clu = pd.read_csv(indir / "category_by_cluster.csv")
    cluster_col = args.cluster_col
    if cluster_col not in cat_clu.columns:
        if "long_cluster" in cat_clu.columns:
            cluster_col = "long_cluster"
        else:
            # fallback: first non-category column
            cluster_col = [c for c in cat_clu.columns if c not in {"category","token_count","cluster_total","share_in_cluster"}][0]

    stacked_share_plot(
        cat_clu, index_col=cluster_col, cat_col="category", share_col="share_in_cluster",
        title="Category share by cluster (codebook-only)",
        outpath=figdir / "category_share_by_cluster.png"
    )

    # --- 3) Verb vs noun split by category (share within POS)
    pos_cat = pd.read_csv(indir / "pos_by_category.csv")
    # Make two charts: within VERB, within NOUN
    for pos in sorted(pos_cat["pos_group_cb"].unique()):
        sub = pos_cat[pos_cat["pos_group_cb"] == pos].copy()
        pivot = sub.pivot_table(index="category", values="share_within_pos", aggfunc="sum", fill_value=0).sort_values("share_within_pos", ascending=False)
        plt.figure(figsize=(10, 5))
        pivot["share_within_pos"].plot(kind="bar")
        plt.title(f"Category share within POS = {pos} (codebook-only)")
        plt.ylabel("Share within POS")
        plt.xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figdir / f"category_share_within_{pos}.png", dpi=200)
        plt.show()

    # --- 4) Agency Index by cluster (nice “one-number” story)
    # Agency Index = (CONTROL + AUTHORSHIP) / (AI_USE + IDEATION)
    # computed on shares within each cluster
    cat_clu_piv = cat_clu.pivot_table(index=cluster_col, columns="category", values="share_in_cluster", aggfunc="sum", fill_value=0)
    num = cat_clu_piv.get("CONTROL", 0) + cat_clu_piv.get("AUTHORSHIP", 0)
    den = cat_clu_piv.get("AI_USE", 0) + cat_clu_piv.get("IDEATION", 0)
    agency = (num / den.replace(0, pd.NA)).fillna(0)
    plt.figure(figsize=(12, 5))
    agency.sort_values(ascending=False).plot(kind="bar")
    plt.title("Agency Index by cluster (CONTROL+AUTHORSHIP)/(AI_USE+IDEATION)")
    plt.ylabel("Agency Index")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "agency_index_by_cluster.png", dpi=200)
    plt.show()

    print("Saved figures to:", figdir)


if __name__ == "__main__":
    main()
