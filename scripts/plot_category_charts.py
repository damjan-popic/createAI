#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def find_col_case_insensitive(df: pd.DataFrame, wanted: set[str]) -> str:
    cols = list(df.columns)
    low = {c.lower().strip(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    # fallback: contains
    for c in cols:
        cl = c.lower().strip()
        if any(w in cl for w in wanted):
            return c
    raise ValueError(f"Could not find any of columns {wanted}. Available: {cols}")


def load_codebook(excel_path: Path, sheet_name: str = "codebook") -> pd.DataFrame:
    """
    Reads the Excel codebook sheet and returns a clean 3-column dataframe:
      lemma, pos_group, category
    The Excel may have CATEGORY (uppercase) — we normalize it to 'category'.
    """
    cb_raw = pd.read_excel(excel_path, sheet_name=sheet_name)

    lemma_col = find_col_case_insensitive(cb_raw, {"lemma"})
    pos_col = find_col_case_insensitive(cb_raw, {"pos_group", "posgroup", "pos"})
    cat_col = find_col_case_insensitive(cb_raw, {"category", "categories", "cat"})

    cb = cb_raw[[lemma_col, pos_col, cat_col]].copy()
    cb.columns = ["lemma", "pos_group", "category"]

    cb["lemma"] = cb["lemma"].astype(str).str.strip()
    cb["pos_group"] = cb["pos_group"].astype(str).str.upper().str.strip()
    cb["category"] = cb["category"].astype(str).str.strip()

    cb.loc[cb["category"].isin(["", "nan", "None"]), "category"] = pd.NA
    cb = cb.dropna(subset=["category"])

    return cb


def prepare_top_table(df: pd.DataFrame, codebook: pd.DataFrame, pos_value: str) -> pd.DataFrame:
    """
    Ensures there's exactly one 'category' column in df, sourced from the Excel codebook.

    Handles all these cases gracefully:
    - df has no category columns -> merge adds 'category'
    - df already has category -> drop it and re-merge from codebook
    - df has category_x/category_y -> drop them and re-merge from codebook
    """
    out = df.copy()

    # Normalize lemma
    if "lemma" not in out.columns:
        raise ValueError(f"Top table missing 'lemma' column. Columns: {list(out.columns)}")
    out["lemma"] = out["lemma"].astype(str).str.strip()

    # Remove any existing category-ish columns to avoid category_x/category_y mess
    drop_cols = [c for c in out.columns if c.lower().startswith("category")]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    # Filter codebook by POS
    cb_pos = codebook[codebook["pos_group"].str.contains(pos_value, na=False)].copy()

    # Merge in fresh category
    out = out.merge(cb_pos[["lemma", "category"]], on="lemma", how="left")

    out["category"] = out["category"].fillna("UNLABELED")
    return out


def plot_stacked(pivot: pd.DataFrame, title: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook_xlsx", required=True, help="Excel with categories (sheet name must be 'codebook')")
    ap.add_argument("--verbs", default="analysis/top100_verbs_with_clusters.csv")
    ap.add_argument("--nouns", default="analysis/top100_nouns_with_clusters.csv")
    ap.add_argument("--cluster_col", default="group", help="Cluster label column (usually 'group' or 'long_cluster')")
    ap.add_argument("--section_col", default="section")
    ap.add_argument("--count_col", default="count")
    ap.add_argument("--outdir", default="analysis/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load codebook (from sheet 'codebook')
    codebook = load_codebook(Path(args.codebook_xlsx), sheet_name="codebook")

    # Load top100 tables
    verbs_raw = pd.read_csv(args.verbs)
    nouns_raw = pd.read_csv(args.nouns)

    # Force categories from codebook (drop any existing category columns in top tables)
    verbs = prepare_top_table(verbs_raw, codebook, "VERB")
    nouns = prepare_top_table(nouns_raw, codebook, "NOUN")

    # Choose cluster column
    cluster_col = args.cluster_col
    if cluster_col not in verbs.columns:
        if "long_cluster" in verbs.columns:
            cluster_col = "long_cluster"
        else:
            raise ValueError(
                f"No cluster column found. Tried '{args.cluster_col}' and 'long_cluster'. "
                f"Available columns: {list(verbs.columns)}"
            )

    # ---- Categories by cluster (verbs vs nouns split)
    v_piv = verbs.pivot_table(
        index=cluster_col, columns="category", values=args.count_col, aggfunc="sum", fill_value=0
    )
    n_piv = nouns.pivot_table(
        index=cluster_col, columns="category", values=args.count_col, aggfunc="sum", fill_value=0
    )

    plot_stacked(
        v_piv,
        title="Verb categories by cluster (weighted by lemma frequency in top-100 lists)",
        ylabel=f"Sum of {args.count_col}",
        outpath=outdir / "verbs_categories_by_cluster.png",
    )
    plot_stacked(
        n_piv,
        title="Noun categories by cluster (weighted by lemma frequency in top-100 lists)",
        ylabel=f"Sum of {args.count_col}",
        outpath=outdir / "nouns_categories_by_cluster.png",
    )

    # ---- Categories by section (verbs vs nouns split)
    if args.section_col not in verbs.columns:
        raise ValueError(
            f"Section column '{args.section_col}' not found in verbs table. "
            f"Available columns: {list(verbs.columns)}"
        )

    v_sec = verbs.pivot_table(
        index=args.section_col, columns="category", values=args.count_col, aggfunc="sum", fill_value=0
    )
    n_sec = nouns.pivot_table(
        index=args.section_col, columns="category", values=args.count_col, aggfunc="sum", fill_value=0
    )

    plot_stacked(
        v_sec,
        title="Verb categories by section (weighted by lemma frequency in top-100 lists)",
        ylabel=f"Sum of {args.count_col}",
        outpath=outdir / "verbs_categories_by_section.png",
    )
    plot_stacked(
        n_sec,
        title="Noun categories by section (weighted by lemma frequency in top-100 lists)",
        ylabel=f"Sum of {args.count_col}",
        outpath=outdir / "nouns_categories_by_section.png",
    )

    print(f"\nSaved figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
