#!/usr/bin/env python3
"""
Print a compact, pasteable report from analysis/triplet_stats outputs.

Usage:
  python scripts/print_triplet_stats_for_chat.py --outdir analysis/triplet_stats
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

SECTIONS_ORDER = ["basic_job_description", "walkthrough", "project_example", "dynamic"]

def read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)

def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x*100:5.1f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/triplet_stats")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)

    cov = read_csv(outdir / "coverage_by_section.csv")
    sec = read_csv(outdir / "label_share_by_section.csv")
    sec_cl = read_csv(outdir / "label_share_by_section_cluster.csv")
    top = read_csv(outdir / "top_triplets_by_section_label.csv")
    occ_path = outdir / "triplet_occurrences.csv"

    print("\n=== Triplet codebook FULL REPORT (PASTE THIS OUTPUT) ===\n")
    print(f"outdir: {outdir.resolve()}\n")

    # 0) Coverage
    print("## 0) Coverage (how much extracted data is labeled by HUMAN/AI)\n")
    if cov.empty:
        print("(missing coverage_by_section.csv)")
    else:
        cov = cov.copy()
        cov["labeled_share"] = cov["labeled_share"].map(pct)
        cov = cov.sort_values("total_occ", ascending=False)
        print(cov.to_string(index=False))
    print()

    # 1) HUMAN vs AI by section
    print("## 1) HUMAN vs AI share by section (labeled only)\n")
    if sec.empty:
        print("(missing label_share_by_section.csv)")
    else:
        sec2 = sec.copy()
        # order sections
        sec2["section"] = pd.Categorical(sec2["section"], categories=SECTIONS_ORDER, ordered=True)
        sec2 = sec2.sort_values(["section","agent_binary"])
        sec2["share"] = sec2["share"].map(pct)
        print(sec2.to_string(index=False))
    print()

    # 2) By section × cluster — show top clusters by volume and biggest AI-shifts
    print("## 2) Section × cluster (job areas): top clusters + biggest shifts\n")
    if sec_cl.empty:
        print("(missing label_share_by_section_cluster.csv)")
    else:
        d = sec_cl.copy()
        d["section"] = pd.Categorical(d["section"], categories=SECTIONS_ORDER, ordered=True)

        # compute volume per section×cluster
        vol = (d.groupby(["section","cluster"])["total"].first().reset_index()
               if "total" in d.columns else
               d.groupby(["section","cluster"])["count"].sum().reset_index().rename(columns={"count":"total"}))

        # overall volume by cluster
        volc = vol.groupby("cluster")["total"].sum().sort_values(ascending=False)
        top_clusters = volc.head(10).index.tolist()

        # AI share table for top clusters
        ai = d[d["agent_binary"]=="AI"].copy()
        ai["share"] = ai["share"].astype(float)
        ai_top = ai[ai["cluster"].isin(top_clusters)].copy()
        ai_top = ai_top.sort_values(["cluster","section"])

        print("### 2a) AI share by section for TOP clusters (by labeled volume)\n")
        if not ai_top.empty:
            tab = ai_top.pivot_table(index="cluster", columns="section", values="share", aggfunc="first", fill_value=np.nan)
            tab = tab.reindex(top_clusters)
            tab = tab.applymap(pct)
            print(tab.to_string())
        else:
            print("(no AI rows?)")
        print()

        # dynamic vs walkthrough delta for all clusters with enough data
        print("### 2b) Dynamic minus Walkthrough (AI share) — clusters with enough data\n")
        ai_wide = ai.pivot_table(index="cluster", columns="section", values="share", aggfunc="first")
        # require both sections present
        if "dynamic" in ai_wide.columns and "walkthrough" in ai_wide.columns:
            ai_wide["delta_dynamic_minus_walkthrough"] = ai_wide["dynamic"] - ai_wide["walkthrough"]
            # add total volume
            ai_wide = ai_wide.merge(volc.rename("total_labeled").reset_index(), on="cluster", how="left")
            # filter small
            ai_wide2 = ai_wide[ai_wide["total_labeled"].fillna(0) >= 30].copy()
            ai_wide2 = ai_wide2.sort_values("delta_dynamic_minus_walkthrough", ascending=False)
            show = ai_wide2[["total_labeled","walkthrough","dynamic","delta_dynamic_minus_walkthrough"]].head(15)
            show = show.rename(columns={
                "walkthrough":"AI_share_walkthrough",
                "dynamic":"AI_share_dynamic",
                "delta_dynamic_minus_walkthrough":"Δ(AI_dynamic - AI_walkthrough)"
            })
            for c in ["AI_share_walkthrough","AI_share_dynamic","Δ(AI_dynamic - AI_walkthrough)"]:
                show[c] = show[c].map(pct)
            show["total_labeled"] = show["total_labeled"].fillna(0).astype(int)
            print(show.to_string())
        else:
            print("(cannot compute deltas: missing dynamic or walkthrough column)")
        print()

    # 3) Top triplets by section × label
    print("## 3) Top triplets by section × label (what drives the differences)\n")
    if top.empty:
        print("(missing top_triplets_by_section_label.csv)")
    else:
        t = top.copy()
        t["section"] = pd.Categorical(t["section"], categories=SECTIONS_ORDER, ordered=True)
        t = t.sort_values(["section","agent_binary","count"], ascending=[True, True, False])

        for secname in SECTIONS_ORDER:
            sub = t[t["section"]==secname]
            if sub.empty:
                continue
            print(f"\n### section = {secname}\n")
            for lab in ["HUMAN","AI"]:
                sub2 = sub[sub["agent_binary"]==lab].head(args.topk)
                if sub2.empty:
                    continue
                print(f"Top {args.topk} — label={lab}")
                print(sub2[["triplet_sig","count"]].to_string(index=False))
                print()

    # 4) Dynamic distinctive triplets (optional, uses occurrences)
    print("\n## 4) Dynamic-distinctive triplets (optional, uses triplet_occurrences.csv)\n")
    if not occ_path.exists():
        print("(no triplet_occurrences.csv found — rerun stats with --write_occurrences to enable this)\n")
        print("\n=== END REPORT ===\n")
        return

    occ = pd.read_csv(occ_path)
    # build signature
    for c in ["subj_lemma","pred_lemma","obj_lemma","action_lemma"]:
        if c in occ.columns:
            occ[c] = occ[c].fillna("").astype(str)
    occ["triplet_sig"] = occ["subj_lemma"] + " :: " + occ["pred_lemma"] + " :: " + occ["obj_lemma"] + occ["action_lemma"].map(lambda x: "" if x=="" else f" → {x}")

    dyn = occ[occ["section"]=="dynamic"]
    oth = occ[occ["section"]!="dynamic"]

    cd = dyn.groupby("triplet_sig").size().rename("count_dynamic")
    co = oth.groupby("triplet_sig").size().rename("count_other")
    tot = pd.concat([cd, co], axis=1).fillna(0).reset_index()

    alpha = 0.5
    Nd = float(len(dyn))
    No = float(len(oth))
    tot["p_dyn"] = (tot["count_dynamic"] + alpha) / (Nd + alpha*2)
    tot["p_oth"] = (tot["count_other"] + alpha) / (No + alpha*2)
    tot["log_odds_dyn"] = np.log(tot["p_dyn"] / tot["p_oth"])
    tot = tot[tot["count_dynamic"] >= 5].sort_values("log_odds_dyn", ascending=False).head(30)

    print("Top 30 distinctive-for-dynamic triplets (count_dynamic>=5):\n")
    print(tot[["triplet_sig","count_dynamic","count_other","log_odds_dyn"]].to_string(index=False))

    print("\n=== END REPORT ===\n")

if __name__ == "__main__":
    main()
