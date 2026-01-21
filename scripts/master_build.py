#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def detect_cluster_cols(cl: pd.DataFrame) -> list[str]:
    keep = ["transcript_id"]
    for cand in ["group", "long_cluster", "cluster", "kmeans_cluster"]:
        if cand in cl.columns:
            keep.append(cand)
    return keep


def load_targets_with_categories(path: Path) -> pd.DataFrame:
    """
    Load lemma-level 'show-off' table with collocators + both category suggestions.
    Expected columns include:
      lemma, pos_group, doc_count, total_count,
      top_subjects, top_objects, top_comps, top_obls,
      top_noun_roles, top_head_verbs, top_amod, top_compound,
      ChatGPT_suggested_category, Claude_suggested_category, Claude_notes
    """
    df = pd.read_csv(path)
    df = norm_cols(df)

    # Normalize pos_group values
    df["pos_group"] = df["pos_group"].astype(str).str.upper().str.strip()
    df.loc[df["pos_group"].str.contains("NOUN", na=False), "pos_group"] = "NOUN"
    df.loc[df["pos_group"].str.contains("VERB", na=False), "pos_group"] = "VERB"

    if "lemma" not in df.columns or "pos_group" not in df.columns:
        raise ValueError("targets_with_categories.csv must contain at least: lemma, pos_group")

    # Fill NaNs for readability
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("")

    return df


def top_clusters_str(dist_df: pd.DataFrame, group_col: str, k: int = 3) -> pd.DataFrame:
    """
    dist_df columns: [group_col, lemma, count]
    Returns: lemma -> "group:count | group:count | ..."
    """
    rows = []
    for lemma, g in dist_df.groupby("lemma"):
        top = g.head(k)
        s = " | ".join([f"{row[group_col]}:{int(row['count'])}" for _, row in top.iterrows()])
        rows.append({"lemma": lemma, "top_clusters": s})
    return pd.DataFrame(rows)


def dedupe_top_clusters(topc: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure exactly one row per lemma in top_clusters table.
    If duplicates exist, keep the longest top_clusters string (usually the richest).
    """
    topc = topc.copy()
    topc["top_clusters"] = topc["top_clusters"].fillna("").astype(str)
    topc["__len"] = topc["top_clusters"].str.len()
    topc = topc.sort_values("__len", ascending=False).drop_duplicates("lemma", keep="first").drop(columns="__len")
    return topc


def main():
    BASE = Path(".")
    OUT = BASE / "analysis"
    ensure_dir(OUT)

    # -------- Inputs
    clusters_path = BASE / "interviews_clustered.tsv"
    transcript_summary_path = BASE / "stanza_out" / "transcript_summary.parquet"
    section_summary_path = BASE / "stanza_out" / "section_summary.parquet"

    # top100 per section (already built earlier)
    top_verbs_path = OUT / "top100_verbs_per_section.csv"
    top_nouns_path = OUT / "top100_nouns_per_section.csv"

    # canonical lemma table for show-off + enrichment
    targets_path = OUT / "targets_with_categories.csv"

    # -------- Load
    cl = norm_cols(read_tsv(clusters_path))
    cl_keep = detect_cluster_cols(cl)
    cl_small = cl[cl_keep].copy()

    trs = pd.read_parquet(transcript_summary_path)
    sec = pd.read_parquet(section_summary_path)

    verbs = pd.read_csv(top_verbs_path)
    nouns = pd.read_csv(top_nouns_path)

    targets = load_targets_with_categories(targets_path)

    # -------- Interview-level
    trs_user = trs[trs["role"] == "user"].copy()
    interview_level = cl_small.merge(trs_user, on="transcript_id", how="left")
    interview_level.to_parquet(OUT / "interview_level_with_clusters.parquet", index=False)
    interview_level.to_csv(OUT / "interview_level_with_clusters.csv", index=False)

    # -------- Section-level
    sec_user = sec[sec["role"] == "user"].copy()
    section_level = sec_user.merge(cl_small, on="transcript_id", how="left")
    section_level.to_parquet(OUT / "section_level_with_clusters.parquet", index=False)
    section_level.to_csv(OUT / "section_level_with_clusters.csv", index=False)

    # -------- Enrich top100 verbs/nouns with clusters + targets lookup
    verbs_join = verbs.merge(cl_small, on="transcript_id", how="left")
    nouns_join = nouns.merge(cl_small, on="transcript_id", how="left")

    verb_lookup = targets[targets["pos_group"] == "VERB"].copy()
    noun_lookup = targets[targets["pos_group"] == "NOUN"].copy()

    verbs_join = verbs_join.merge(verb_lookup, on="lemma", how="left", suffixes=("", "_lex"))
    nouns_join = nouns_join.merge(noun_lookup, on="lemma", how="left", suffixes=("", "_lex"))

    verbs_join.to_parquet(OUT / "top100_verbs_with_clusters.parquet", index=False)
    verbs_join.to_csv(OUT / "top100_verbs_with_clusters.csv", index=False)

    nouns_join.to_parquet(OUT / "top100_nouns_with_clusters.parquet", index=False)
    nouns_join.to_csv(OUT / "top100_nouns_with_clusters.csv", index=False)

    # -------- SHOW-OFF / QUICK GLANCE file
    # Add a human-readable "top_clusters" hint per lemma:
    # which clusters most frequently use this lemma (based on top100 tables).
    gcol = None
    if "group" in cl_small.columns:
        gcol = "group"
    elif "long_cluster" in cl_small.columns:
        gcol = "long_cluster"

    if gcol:
        verb_dist = (
            verbs_join.groupby([gcol, "lemma"], as_index=False)["count"].sum()
            .sort_values(["lemma", "count"], ascending=[True, False])
        )
        noun_dist = (
            nouns_join.groupby([gcol, "lemma"], as_index=False)["count"].sum()
            .sort_values(["lemma", "count"], ascending=[True, False])
        )

        verb_topc = top_clusters_str(verb_dist, gcol, k=3)
        noun_topc = top_clusters_str(noun_dist, gcol, k=3)

        topc = dedupe_top_clusters(pd.concat([verb_topc, noun_topc], ignore_index=True))

        show = targets.merge(topc, on="lemma", how="left")
        show["top_clusters"] = show["top_clusters"].fillna("")
    else:
        show = targets.copy()
        show["top_clusters"] = ""

    # Make it “glanceable”
    col_order = [
        "lemma", "pos_group", "doc_count", "total_count",
        "ChatGPT_suggested_category", "Claude_suggested_category", "Claude_notes",
        "top_clusters",
        "top_subjects", "top_objects", "top_comps", "top_obls",
        "top_noun_roles", "top_head_verbs", "top_amod", "top_compound",
    ]
    for c in col_order:
        if c not in show.columns:
            show[c] = ""
    show = show[col_order]

    # Sort: first by POS, then by doc_count/total_count
    show = show.sort_values(["pos_group", "doc_count", "total_count", "lemma"], ascending=[True, False, False, True])
    show.to_csv(OUT / "showoff_targets_with_clusters.csv", index=False)

    # -------- Cluster summaries (quick “see something”)
    if gcol:
        cluster_sizes = cl_small[gcol].value_counts().reset_index()
        cluster_sizes.columns = [gcol, "n_interviews"]
        cluster_sizes.to_csv(OUT / "cluster_sizes.csv", index=False)

        top_verbs_cluster = (
            verbs_join.groupby([gcol, "lemma"], as_index=False)["count"].sum()
            .sort_values([gcol, "count"], ascending=[True, False])
            .groupby(gcol).head(25)
        )
        top_verbs_cluster.to_csv(OUT / "top_verbs_by_cluster.csv", index=False)

        top_nouns_cluster = (
            nouns_join.groupby([gcol, "lemma"], as_index=False)["count"].sum()
            .sort_values([gcol, "count"], ascending=[True, False])
            .groupby(gcol).head(25)
        )
        top_nouns_cluster.to_csv(OUT / "top_nouns_by_cluster.csv", index=False)

    print("Wrote master outputs to analysis/:")
    print(" - interview_level_with_clusters.(parquet|csv)")
    print(" - section_level_with_clusters.(parquet|csv)")
    print(" - top100_verbs_with_clusters.(parquet|csv)")
    print(" - top100_nouns_with_clusters.(parquet|csv)")
    print(" - showoff_targets_with_clusters.csv")
    print(" - cluster_sizes.csv, top_verbs_by_cluster.csv, top_nouns_by_cluster.csv (if cluster column exists)")


if __name__ == "__main__":
    main()
