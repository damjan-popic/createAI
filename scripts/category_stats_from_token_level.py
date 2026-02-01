#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_codebook(xlsx: Path, sheet: str = "codebook") -> pd.DataFrame:
    cb = pd.read_excel(xlsx, sheet_name=sheet).copy()
    cols = {c.strip().lower(): c for c in cb.columns}

    lemma_col = cols.get("lemma")
    pos_col = cols.get("pos_group") or cols.get("posgroup") or cols.get("pos")
    cat_col = cols.get("category") or cols.get("categories") or cols.get("cat")

    if not lemma_col or not pos_col or not cat_col:
        raise ValueError(f"Codebook must have lemma, pos_group, CATEGORY. Found: {list(cb.columns)}")

    cb = cb[[lemma_col, pos_col, cat_col]].copy()
    cb.columns = ["lemma", "pos_group", "category"]

    cb["lemma"] = cb["lemma"].astype(str).str.strip().str.lower()
    cb["pos_group"] = cb["pos_group"].astype(str).str.upper().str.strip()
    cb["category"] = cb["category"].astype(str).str.upper().str.strip()

    cb = cb[cb["category"].notna() & (cb["category"] != "")]

    # normalize POS for your split
    cb.loc[cb["pos_group"].str.contains("VERB", na=False), "pos_group"] = "VERB"
    cb.loc[cb["pos_group"].str.contains("NOUN", na=False), "pos_group"] = "NOUN"
    cb.loc[cb["pos_group"].str.contains("PROPN", na=False), "pos_group"] = "NOUN"  # treat PROPN as NOUN

    return cb


def read_clusters(tsv: Path) -> pd.DataFrame:
    cl = pd.read_csv(tsv, sep="\t").copy()
    if "transcript_id" not in cl.columns:
        raise ValueError(f"clusters file must contain transcript_id. Found: {list(cl.columns)}")

    keep = ["transcript_id"]
    if "group" in cl.columns:
        keep.append("group")
    if "long_cluster" in cl.columns:
        keep.append("long_cluster")

    return cl[keep].copy()


def ensure_token_schema(tok: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["transcript_id", "section", "lemma"] if c not in tok.columns]
    if missing:
        raise ValueError(f"token_level.parquet missing columns {missing}. Found: {list(tok.columns)}")

    tok = tok.copy()
    tok["lemma"] = tok["lemma"].astype(str).str.strip().str.lower()

    if "role" in tok.columns:
        tok["role"] = tok["role"].astype(str).str.lower().str.strip()

    if "pos_group" not in tok.columns:
        if "upos" in tok.columns:
            tok["pos_group"] = tok["upos"]
        else:
            tok["pos_group"] = "UNK"

    tok["pos_group"] = tok["pos_group"].astype(str).str.upper().str.strip()
    tok.loc[tok["pos_group"].str.contains("PROPN", na=False), "pos_group"] = "NOUN"
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_parquet", default="stanza_out/token_level.parquet")
    ap.add_argument("--codebook_xlsx", default="analysis/targets_with_categories.xlsx")
    ap.add_argument("--codebook_sheet", default="codebook")
    ap.add_argument("--clusters_tsv", default="interviews_clustered.tsv")
    ap.add_argument("--outdir", default="analysis/categorized")
    ap.add_argument("--role", default="user", help="Use 'user' to exclude assistant tokens. Use 'all' to include everything.")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cb = read_codebook(Path(args.codebook_xlsx), sheet=args.codebook_sheet)
    cl = read_clusters(Path(args.clusters_tsv))

    tok = pd.read_parquet(args.token_parquet)
    tok = ensure_token_schema(tok)

    if args.role != "all" and "role" in tok.columns:
        tok = tok[tok["role"] == args.role]

    # === Restrict universe to codebook lemmas ONLY ===
    tok = tok.merge(cb, on="lemma", how="inner")  # adds pos_group_y (codebook) + category
    if "pos_group_y" in tok.columns:
        tok = tok.rename(columns={"pos_group_y": "pos_group_cb"})
    else:
        tok["pos_group_cb"] = tok["pos_group"]

    tok["pos_group_cb"] = tok["pos_group_cb"].astype(str).str.upper().str.strip()
    tok = tok.merge(cl, on="transcript_id", how="left")

    cluster_col = "group" if "group" in tok.columns else ("long_cluster" if "long_cluster" in tok.columns else None)
    if cluster_col is None:
        tok["group"] = "UNKNOWN"
        cluster_col = "group"

    tok["token_count"] = 1

    # 1) category-by-section (counts + shares)
    cat_sec = tok.groupby(["section", "category"], as_index=False)["token_count"].sum()
    sec_tot = tok.groupby(["section"], as_index=False)["token_count"].sum().rename(columns={"token_count": "section_total"})
    cat_sec = cat_sec.merge(sec_tot, on="section", how="left")
    cat_sec["share_in_section"] = cat_sec["token_count"] / cat_sec["section_total"]
    cat_sec.to_csv(outdir / "category_by_section.csv", index=False)

    # 2) category-by-cluster (counts + shares)
    cat_clu = tok.groupby([cluster_col, "category"], as_index=False)["token_count"].sum()
    clu_tot = tok.groupby([cluster_col], as_index=False)["token_count"].sum().rename(columns={"token_count": "cluster_total"})
    cat_clu = cat_clu.merge(clu_tot, on=cluster_col, how="left")
    cat_clu["share_in_cluster"] = cat_clu["token_count"] / cat_clu["cluster_total"]
    cat_clu.to_csv(outdir / "category_by_cluster.csv", index=False)

    # 3) verb vs noun split by category (counts + shares)
    pos_cat = tok.groupby(["pos_group_cb", "category"], as_index=False)["token_count"].sum()
    pos_tot = tok.groupby(["pos_group_cb"], as_index=False)["token_count"].sum().rename(columns={"token_count": "pos_total"})
    pos_cat = pos_cat.merge(pos_tot, on="pos_group_cb", how="left")
    pos_cat["share_within_pos"] = pos_cat["token_count"] / pos_cat["pos_total"]
    pos_cat.to_csv(outdir / "pos_by_category.csv", index=False)

    # 4) Which lemmas dominate each category (overall)
    lemma_cat = (
        tok.groupby(["category", "pos_group_cb", "lemma"], as_index=False)["token_count"].sum()
        .sort_values(["category", "pos_group_cb", "token_count"], ascending=[True, True, False])
    )
    lemma_cat.to_csv(outdir / "lemma_by_category_full.csv", index=False)
    lemma_cat.groupby(["category", "pos_group_cb"]).head(args.topk).to_csv(outdir / "top_lemmas_by_category.csv", index=False)

    # 5) Top lemmas by category BY CLUSTER
    lemma_cat_cluster = (
        tok.groupby([cluster_col, "category", "pos_group_cb", "lemma"], as_index=False)["token_count"].sum()
        .sort_values([cluster_col, "category", "pos_group_cb", "token_count"], ascending=[True, True, True, False])
    )
    lemma_cat_cluster.to_csv(outdir / "lemma_by_category_by_cluster_full.csv", index=False)
    lemma_cat_cluster.groupby([cluster_col, "category", "pos_group_cb"]).head(min(10, args.topk)).to_csv(
        outdir / "top_lemmas_by_category_by_cluster.csv", index=False
    )

    # 6) Top lemmas by category BY SECTION
    lemma_cat_section = (
        tok.groupby(["section", "category", "pos_group_cb", "lemma"], as_index=False)["token_count"].sum()
        .sort_values(["section", "category", "pos_group_cb", "token_count"], ascending=[True, True, True, False])
    )
    lemma_cat_section.to_csv(outdir / "lemma_by_category_by_section_full.csv", index=False)
    lemma_cat_section.groupby(["section", "category", "pos_group_cb"]).head(min(10, args.topk)).to_csv(
        outdir / "top_lemmas_by_category_by_section.csv", index=False
    )

    # 7) Convenience: category totals (overall)
    cat_tot = tok.groupby(["category"], as_index=False)["token_count"].sum().sort_values("token_count", ascending=False)
    cat_tot.to_csv(outdir / "category_totals.csv", index=False)

    print("Wrote outputs to:", outdir)


if __name__ == "__main__":
    main()
