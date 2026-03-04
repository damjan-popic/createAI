#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_ORDER = ["basic_job_description", "walkthrough", "project_example", "dynamic"]


def pick_cluster_col(df: pd.DataFrame) -> str:
    """Prefer 'group' if it has signal; else use 'long_cluster'; else fallback."""
    for col in ["group", "long_cluster"]:
        if col in df.columns and df[col].fillna("").astype(str).str.strip().ne("").any():
            return col
    # last resort
    return "group" if "group" in df.columns else ("long_cluster" if "long_cluster" in df.columns else "")


def ensure_section_order(piv: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    cols = [c for c in order if c in piv.columns]
    missing = [c for c in order if c not in piv.columns]
    # add missing columns as zeros
    for c in missing:
        piv[c] = 0.0
        cols.append(c)
    return piv[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_dir", default="analysis/final/triplet_stats")
    ap.add_argument("--sections_order", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--topn_clusters", type=int, default=12)
    ap.add_argument("--topn_creative_types", type=int, default=12)
    args = ap.parse_args()

    final_dir = Path(args.final_dir)
    figdir = final_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    sections_order = [s.strip() for s in args.sections_order.split(",") if s.strip()]

    occ_path = final_dir / "triplet_occurrences_labeled.csv"
    if not occ_path.exists():
        raise SystemExit(
            f"Missing {occ_path}. Re-run triplet_full_stats.py with --write_occurrences "
            "so we have the labeled occurrences to plot from."
        )

    df = pd.read_csv(occ_path)

    # Keep only your 4 sections (and force the order)
    df["section"] = df["section"].astype(str)
    df = df[df["section"].isin(sections_order)].copy()
    df["section"] = pd.Categorical(df["section"], categories=sections_order, ordered=True)

    # Coverage plot needs all rows (incl unlabeled). Labeled-only plots filter here:
    df["agent_binary"] = df["agent_binary"].fillna("").astype(str)
    df["is_labeled"] = df["agent_binary"].isin(["HUMAN", "AI"])

    # -------------------------
    # 1) coverage_labeled_share_by_section.png
    # -------------------------
    cov = (df.groupby("section")
             .agg(total_occ=("section", "size"),
                  labeled_occ=("is_labeled", "sum")))
    cov["labeled_share"] = cov["labeled_occ"] / cov["total_occ"].replace(0, pd.NA)
    cov = cov.reindex(sections_order)

    ax = cov[["labeled_share"]].plot(kind="bar", figsize=(7, 3))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share labeled (HUMAN/AI)")
    ax.set_title("Coverage: labeled share of extracted triplets")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "coverage_labeled_share_by_section.png", dpi=220)
    plt.close()

    # Work with labeled subset from here on
    lab = df[df["is_labeled"]].copy()
    lab["one"] = 1

    # -------------------------
    # 2) label_share_by_section.png (stacked HUMAN/AI)
    # -------------------------
    s1 = (lab.groupby(["section", "agent_binary"])["one"].sum()
            .rename("count").reset_index())
    totals = s1.groupby("section")["count"].sum().rename("total").reset_index()
    s1 = s1.merge(totals, on="section", how="left")
    s1["share"] = s1["count"] / s1["total"]

    piv = s1.pivot_table(index="section", columns="agent_binary", values="share", aggfunc="sum", fill_value=0)
    for col in ["HUMAN", "AI"]:
        if col not in piv.columns:
            piv[col] = 0.0
    piv = piv[["HUMAN", "AI"]].reindex(sections_order)

    ax = piv.plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share among labeled triplets")
    ax.set_title("HUMAN vs AI share by section (triplet codebook)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "label_share_by_section.png", dpi=220)
    plt.close()

    # -------------------------
    # 3) ai_share_heatmap_top_clusters.png
    # -------------------------
    cluster_col = pick_cluster_col(lab)
    if cluster_col:
        vol = lab.groupby(cluster_col).size().sort_values(ascending=False)
        top_clusters = vol.head(args.topn_clusters).index.tolist()

        ai = (lab.groupby(["section", cluster_col, "agent_binary"])["one"].sum()
                .rename("count").reset_index())
        tot = ai.groupby(["section", cluster_col])["count"].sum().rename("total").reset_index()
        ai = ai.merge(tot, on=["section", cluster_col], how="left")
        ai["share"] = ai["count"] / ai["total"]
        ai = ai[ai["agent_binary"] == "AI"].copy()

        heat = ai[ai[cluster_col].isin(top_clusters)].pivot_table(
            index=cluster_col, columns="section", values="share", aggfunc="first", fill_value=0
        )
        heat = heat.reindex(top_clusters)
        heat = ensure_section_order(heat, sections_order)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(heat.values, aspect="auto")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns), rotation=20, ha="right")
        ax.set_title(f"AI share by {cluster_col} × section (top {args.topn_clusters})")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()
        plt.savefig(figdir / "ai_share_heatmap_top_clusters.png", dpi=220)
        plt.close()
    else:
        print("Warning: no cluster columns found (group/long_cluster). Skipping top_clusters heatmap.")

    # -------------------------
    # 4) ai_share_heatmap_top_creative_types.png
    # -------------------------
    if "creative_type" in lab.columns:
        vol = lab.groupby("creative_type").size().sort_values(ascending=False)
        top_types = vol.head(args.topn_creative_types).index.tolist()

        ai = (lab.groupby(["section", "creative_type", "agent_binary"])["one"].sum()
                .rename("count").reset_index())
        tot = ai.groupby(["section", "creative_type"])["count"].sum().rename("total").reset_index()
        ai = ai.merge(tot, on=["section", "creative_type"], how="left")
        ai["share"] = ai["count"] / ai["total"]
        ai = ai[ai["agent_binary"] == "AI"].copy()

        heat = ai[ai["creative_type"].isin(top_types)].pivot_table(
            index="creative_type", columns="section", values="share", aggfunc="first", fill_value=0
        )
        heat = heat.reindex(top_types)
        heat = ensure_section_order(heat, sections_order)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(heat.values, aspect="auto")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns), rotation=20, ha="right")
        ax.set_title(f"AI share by creative_type × section (top {args.topn_creative_types})")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.tight_layout()
        plt.savefig(figdir / "ai_share_heatmap_top_creative_types.png", dpi=220)
        plt.close()
    else:
        print("Warning: creative_type column not present. Skipping creative_type heatmap.")

    print("✅ Rewrote plots in:", figdir.resolve())


if __name__ == "__main__":
    main()