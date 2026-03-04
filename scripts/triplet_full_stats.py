#!/usr/bin/env python3
"""
Triplet ground-truth stats (HUMAN vs AI) for 4 sections + job areas + per-interview.

Works with:
- stanza_out/token_level.parquet   (token-level UD)
- interviews_clustered.tsv         (transcript_id -> group/long_cluster)
- triplet codebook CSV             (subj_lemma,pred_lemma,obj_lemma,action_lemma,pattern,voice,agent_binary)

Key outputs written under: analysis/final/triplet_stats/ (by default)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import pandas as pd
import numpy as np

try:
    import pyarrow.dataset as ds
except Exception:
    ds = None

# UD relation sets
SUBJ_ACTIVE = {"nsubj", "csubj", "nsubj:outer", "csubj:outer"}
SUBJ_PASS   = {"nsubj:pass", "csubj:pass"}
AGENT_PASS  = {"obl:agent", "agent"}
OBJ_DEPS    = {"obj", "iobj"}
COMP_DEPS   = {"xcomp", "ccomp"}

SPAN_DEPS = {"det","amod","compound","nummod","nmod:poss","nmod","case","fixed","flat","name","neg"}
DETERMINERS = {"the","a","an"}

SECTIONS_DEFAULT = "basic_job_description,walkthrough,project_example,dynamic"

def norm(x) -> str:
    return "" if x is None else str(x).strip()

def norm_lower(x) -> str:
    return norm(x).lower()

def load_clusters(clusters_tsv: Optional[Path]) -> pd.DataFrame:
    if not clusters_tsv:
        return pd.DataFrame({"transcript_id": [], "group": [], "long_cluster": []})
    cl = pd.read_csv(clusters_tsv, sep="\t", dtype=str)
    if "group" not in cl.columns:
        cl["group"] = ""
    if "long_cluster" not in cl.columns:
        cl["long_cluster"] = ""
    return cl[["transcript_id","group","long_cluster"]].drop_duplicates()

def load_codebook(codebook_csv: Path) -> pd.DataFrame:
    cb = pd.read_csv(codebook_csv)
    req = ["subj_lemma","pred_lemma","obj_lemma","action_lemma","pattern","voice","agent_binary"]
    for c in req:
        if c not in cb.columns:
            raise ValueError(f"Codebook missing {c}. Found: {list(cb.columns)}")
    for c in ["subj_lemma","pred_lemma","obj_lemma","action_lemma","pattern","voice"]:
        cb[c] = cb[c].fillna("").astype(str)
    cb["agent_binary"] = cb["agent_binary"].fillna("").astype(str)
    cb = cb.drop_duplicates(subset=["subj_lemma","pred_lemma","obj_lemma","action_lemma","pattern","voice"], keep="first")
    return cb

def load_tokens(token_parquet: Path, sections: List[str], role: str) -> pd.DataFrame:
    if ds is not None:
        dataset = ds.dataset(str(token_parquet), format="parquet")
        filt = ds.field("section").isin(sections)
        if role != "all":
            filt = filt & (ds.field("role") == role)
        return dataset.to_table(filter=filt).to_pandas()

    df = pd.read_parquet(token_parquet)
    df = df[df["section"].isin(sections)]
    if role != "all":
        df = df[df["role"] == role]
    return df

def build_children_map(sent_df: pd.DataFrame) -> Dict[int, List[int]]:
    ch: Dict[int, List[int]] = {}
    for r in sent_df.itertuples(index=False):
        h = int(r.head)
        wid = int(r.word_id)
        if h <= 0:
            continue
        ch.setdefault(h, []).append(wid)
    return ch

def token_by_id(sent_df: pd.DataFrame) -> Dict[int, dict]:
    d = {}
    for r in sent_df.itertuples(index=False):
        d[int(r.word_id)] = {"word_id": int(r.word_id), "text": r.text, "lemma": r.lemma,
                             "upos": r.upos, "deprel": r.deprel, "head": int(r.head)}
    return d

def span_for_head(head_id: int, tok: Dict[int, dict], children: Dict[int, List[int]], max_len: int = 10) -> str:
    ids = [head_id]
    for cid in children.get(head_id, []):
        dep = tok.get(cid)
        if dep and dep["deprel"] in SPAN_DEPS:
            ids.append(cid)
    ids = sorted(set(ids))[:max_len]
    return " ".join([str(tok[i]["text"]) for i in ids if i in tok]).strip()

def repair_determiner_object(obj_id: int, tok: Dict[int, dict]) -> int:
    t = tok.get(obj_id)
    if not t:
        return obj_id
    if t["upos"] == "DET" or norm_lower(t["lemma"]) in DETERMINERS or norm_lower(t["text"]) in DETERMINERS:
        nxt = obj_id + 1
        if nxt in tok:
            return nxt
    return obj_id

def pick_dependents(children: Dict[int, List[int]], tok: Dict[int, dict], head_id: int, rels: set) -> List[int]:
    out = []
    for cid in children.get(head_id, []):
        dep = tok.get(cid)
        if dep and dep["deprel"] in rels:
            out.append(cid)
    return out

def extract_occurrences(tokens: pd.DataFrame, clusters: pd.DataFrame, include_complements: bool) -> pd.DataFrame:
    df = tokens.merge(clusters, on="transcript_id", how="left")
    df["group"] = df["group"].fillna("")
    df["long_cluster"] = df["long_cluster"].fillna("")
    df["word_id"] = pd.to_numeric(df["word_id"], errors="coerce").astype("Int64")
    df["head"] = pd.to_numeric(df["head"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["transcript_id","sent_id","section","role","word_id","lemma","text","upos","deprel","head"])

    rows = []
    gcols = ["transcript_id","sent_id","section","role","group","long_cluster"]
    for (tid, sid, sec, role, grp, lcl), sent in df.groupby(gcols, sort=False):
        sent = sent.sort_values("word_id")
        children = build_children_map(sent)
        tok = token_by_id(sent)

        for v in sent.itertuples(index=False):
            if v.upos not in ("VERB","AUX"):
                continue
            v_id = int(v.word_id)

            passive_agents = pick_dependents(children, tok, v_id, AGENT_PASS)
            active_subjs   = pick_dependents(children, tok, v_id, SUBJ_ACTIVE)
            passive_subjs  = pick_dependents(children, tok, v_id, SUBJ_PASS)
            objects        = pick_dependents(children, tok, v_id, OBJ_DEPS)

            comps = []
            if include_complements:
                comp_ids = pick_dependents(children, tok, v_id, COMP_DEPS)
                for cid in comp_ids:
                    c = tok.get(cid)
                    if c and c.get("upos") in ("VERB","AUX"):
                        comps.append(cid)

            voice = "ACT"
            if passive_agents:
                voice = "PASS"; subj_ids = passive_agents
            elif active_subjs:
                subj_ids = active_subjs
            elif passive_subjs:
                voice = "PASS"; subj_ids = passive_subjs
            else:
                continue

            obj_ids = objects
            if not obj_ids and passive_subjs:
                obj_ids = passive_subjs

            for s_id in subj_ids[:3]:
                s = tok.get(s_id, {})
                subj_lemma = str(s.get("lemma",""))
                subj_text = span_for_head(s_id, tok, children) or str(s.get("text",""))

                # SV
                if not obj_ids and not comps:
                    rows.append(dict(transcript_id=tid, sent_id=int(sid), section=sec, role=role,
                                     cluster=grp, long_cluster=lcl, voice=voice, pattern="SV",
                                     subj_lemma=subj_lemma, pred_lemma=str(v.lemma),
                                     obj_lemma="", action_lemma="",
                                     subj_text=subj_text, pred_text=str(v.text),
                                     obj_text="", action_text=""))
                    continue

                # SVO and/or xcomp
                for o_id in (obj_ids[:3] if obj_ids else [None]):
                    obj_lemma = obj_text = ""
                    if o_id is not None:
                        o_id2 = repair_determiner_object(int(o_id), tok)
                        o = tok.get(o_id2, {})
                        obj_lemma = str(o.get("lemma",""))
                        obj_text = span_for_head(o_id2, tok, children) or str(o.get("text",""))

                    if comps:
                        for c_id in comps[:2]:
                            c = tok.get(c_id, {})
                            rows.append(dict(transcript_id=tid, sent_id=int(sid), section=sec, role=role,
                                             cluster=grp, long_cluster=lcl, voice=voice,
                                             pattern=("SVOxCOMP" if obj_lemma else "SVxCOMP"),
                                             subj_lemma=subj_lemma, pred_lemma=str(v.lemma),
                                             obj_lemma=obj_lemma, action_lemma=str(c.get("lemma","")),
                                             subj_text=subj_text, pred_text=str(v.text),
                                             obj_text=obj_text, action_text=str(c.get("text",""))))
                    else:
                        rows.append(dict(transcript_id=tid, sent_id=int(sid), section=sec, role=role,
                                         cluster=grp, long_cluster=lcl, voice=voice,
                                         pattern=("SVO" if obj_lemma else "SV"),
                                         subj_lemma=subj_lemma, pred_lemma=str(v.lemma),
                                         obj_lemma=obj_lemma, action_lemma="",
                                         subj_text=subj_text, pred_text=str(v.text),
                                         obj_text=obj_text, action_text=""))

    occ = pd.DataFrame(rows)
    for c in ["subj_lemma","pred_lemma","obj_lemma","action_lemma","pattern","voice"]:
        if c in occ.columns:
            occ[c] = occ[c].fillna("").astype(str)
    return occ

def df_map(df: pd.DataFrame, func):
    # pandas 3: DataFrame.map exists; older: use applymap
    if hasattr(df, "map"):
        return df.map(func)
    return df.applymap(func)

def pct(x):
    if pd.isna(x): return ""
    return f"{x*100:5.1f}%"

def write_final(outdir: Path, occ_lab: pd.DataFrame, topk: int, per_transcript_topk: int):
    final = outdir
    # folders
    p00 = final / "00_coverage"; p00.mkdir(parents=True, exist_ok=True)
    p01 = final / "01_section_level"; p01.mkdir(parents=True, exist_ok=True)
    p02 = final / "02_section_cluster_jobareas"; p02.mkdir(parents=True, exist_ok=True)
    p03 = final / "03_overall_triplets"; p03.mkdir(parents=True, exist_ok=True)
    p04 = final / "04_per_interview"; p04.mkdir(parents=True, exist_ok=True)
    fig = final / "figures"; fig.mkdir(parents=True, exist_ok=True)

    # coverage
    cov = (occ_lab.assign(is_labeled=occ_lab["agent_binary"].isin(["HUMAN","AI"]))
           .groupby("section")
           .agg(total_occ=("section","size"),
                labeled_occ=("is_labeled","sum"),
                labeled_share=("is_labeled","mean"))
           .reset_index().sort_values("total_occ", ascending=False))
    cov.to_csv(p00 / "coverage_by_section.csv", index=False)

    labeled = occ_lab[occ_lab["agent_binary"].isin(["HUMAN","AI"])].copy()
    labeled["one"] = 1

    # label share by section
    s1 = (labeled.groupby(["section","agent_binary"])["one"].sum().rename("count").reset_index())
    totals = s1.groupby("section")["count"].sum().rename("total").reset_index()
    s1 = s1.merge(totals, on="section", how="left")
    s1["share"] = s1["count"] / s1["total"]
    s1.to_csv(p01 / "label_share_by_section.csv", index=False)

    # label share by section × cluster
    s2 = (labeled.groupby(["section","cluster","agent_binary"])["one"].sum().rename("count").reset_index())
    totals2 = s2.groupby(["section","cluster"])["count"].sum().rename("total").reset_index()
    s2 = s2.merge(totals2, on=["section","cluster"], how="left")
    s2["share"] = s2["count"] / s2["total"]
    s2.to_csv(p02 / "label_share_by_section_cluster.csv", index=False)

    # label share by cluster overall
    s3 = (labeled.groupby(["cluster","agent_binary"])["one"].sum().rename("count").reset_index())
    totals3 = s3.groupby("cluster")["count"].sum().rename("total").reset_index()
    s3 = s3.merge(totals3, on="cluster", how="left")
    s3["share"] = s3["count"] / s3["total"]
    s3.to_csv(p02 / "label_share_by_cluster.csv", index=False)

    # triplet signature
    labeled["triplet_sig"] = (
        labeled["subj_lemma"] + " :: " + labeled["pred_lemma"] + " :: " + labeled["obj_lemma"] +
        labeled["action_lemma"].map(lambda x: "" if x=="" else f" → {x}")
    )

    # top triplets overall by label
    top_overall = (labeled.groupby(["agent_binary","triplet_sig"]).size()
                   .rename("count").reset_index()
                   .sort_values(["agent_binary","count"], ascending=[True, False]))
    top_overall.head(topk*2).to_csv(p03 / f"top_triplets_overall_label_top{topk*2}.csv", index=False)

    # top triplets by section × label (topk each)
    top_sec = (labeled.groupby(["section","agent_binary","triplet_sig"]).size()
               .rename("count").reset_index()
               .sort_values(["section","agent_binary","count"], ascending=[True, True, False]))
    # keep topk per section×label
    top_sec2 = top_sec.groupby(["section","agent_binary"], as_index=False).head(topk)
    top_sec2.to_csv(p01 / f"top_triplets_by_section_label_top{topk}.csv", index=False)

    # Per interview (overall)
    ti = (labeled.groupby(["transcript_id","cluster","long_cluster","agent_binary"]).size()
          .rename("count").reset_index())
    ti_tot = ti.groupby(["transcript_id","cluster","long_cluster"])["count"].sum().rename("total").reset_index()
    ti = ti.merge(ti_tot, on=["transcript_id","cluster","long_cluster"], how="left")
    ti["share"] = ti["count"] / ti["total"]
    ti.to_csv(p04 / "label_share_by_transcript.csv", index=False)

    # Per interview × section (long)
    tis = (labeled.groupby(["transcript_id","cluster","long_cluster","section","agent_binary"]).size()
           .rename("count").reset_index())
    tis_tot = tis.groupby(["transcript_id","cluster","long_cluster","section"])["count"].sum().rename("total").reset_index()
    tis = tis.merge(tis_tot, on=["transcript_id","cluster","long_cluster","section"], how="left")
    tis["share"] = tis["count"] / tis["total"]
    tis.to_csv(p04 / "label_share_by_transcript_section_long.csv", index=False)

    # Per interview × section (wide) — AI share columns + totals
    ai_only = tis[tis["agent_binary"]=="AI"].copy()
    wide = ai_only.pivot_table(index=["transcript_id","cluster","long_cluster"], columns="section", values="share", aggfunc="first")
    wide = wide.rename(columns={c: f"AI_share_{c}" for c in wide.columns}).reset_index()

    # totals per section (labeled)
    totals_wide = tis_tot.pivot_table(index=["transcript_id","cluster","long_cluster"], columns="section", values="total", aggfunc="first")
    totals_wide = totals_wide.rename(columns={c: f"labeled_total_{c}" for c in totals_wide.columns}).reset_index()

    wide = wide.merge(totals_wide, on=["transcript_id","cluster","long_cluster"], how="left")
    wide.to_csv(p04 / "label_share_by_transcript_section_wide.csv", index=False)

    # Optional: per transcript top triplets in dynamic (topN per transcript × label)
    dyn = labeled[labeled["section"]=="dynamic"].copy()
    if not dyn.empty and per_transcript_topk > 0:
        ttd = (dyn.groupby(["transcript_id","agent_binary","triplet_sig"]).size()
               .rename("count").reset_index()
               .sort_values(["transcript_id","agent_binary","count"], ascending=[True, True, False]))
        ttd = ttd.groupby(["transcript_id","agent_binary"], as_index=False).head(per_transcript_topk)
        ttd.to_csv(p04 / f"top_triplets_dynamic_by_transcript_top{per_transcript_topk}.csv", index=False)

    # Plots
    import matplotlib.pyplot as plt

    piv = s1.pivot_table(index="section", columns="agent_binary", values="share", aggfunc="sum", fill_value=0)
    for col in ["HUMAN","AI"]:
        if col not in piv.columns: piv[col] = 0.0
    piv = piv[["HUMAN","AI"]]
    ax = piv.plot(kind="bar", stacked=True, figsize=(8,4))
    ax.set_ylim(0,1); ax.set_ylabel("Share among labeled triplets")
    ax.set_title("HUMAN vs AI share by section (triplet codebook)")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(fig / "label_share_by_section.png", dpi=220); plt.close()

    # heatmap top clusters
    volc = labeled.groupby("cluster").size().sort_values(ascending=False)
    top_clusters = volc.head(12).index.tolist()
    ai = s2[s2["agent_binary"]=="AI"].copy()
    heat = ai[ai["cluster"].isin(top_clusters)].pivot_table(index="cluster", columns="section", values="share", aggfunc="first", fill_value=0)
    heat = heat.loc[top_clusters]
    fig2, ax2 = plt.subplots(figsize=(10,5))
    im = ax2.imshow(heat.values, aspect="auto")
    ax2.set_yticks(range(len(heat.index))); ax2.set_yticklabels(heat.index)
    ax2.set_xticks(range(len(heat.columns))); ax2.set_xticklabels(heat.columns, rotation=20, ha="right")
    ax2.set_title("AI share by cluster × section (top clusters)")
    fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(fig / "ai_share_heatmap_top_clusters.png", dpi=220); plt.close()

    # coverage plot
    cov2 = cov.set_index("section")[["labeled_share"]]
    ax = cov2.plot(kind="bar", figsize=(7,3))
    ax.set_ylim(0,1); ax.set_ylabel("Share labeled (HUMAN/AI)")
    ax.set_title("Coverage: labeled share of extracted triplets")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(fig / "coverage_labeled_share_by_section.png", dpi=220); plt.close()

    # README
    readme = final / "README.txt"
    readme.write_text(
        "Triplet stats (HUMAN vs AI) — output pack\n\n"
        "00_coverage: how much of extracted triplets are labeled\n"
        "01_section_level: main section-level shares + top triplets per section/label\n"
        "02_section_cluster_jobareas: job-area (cluster) breakdown + heatmap\n"
        "03_overall_triplets: overall most frequent triplets by label\n"
        "04_per_interview: per transcript AI/HUMAN shares (overall + per section) in long+wide form\n"
        "figures: ready-to-share PNGs\n", encoding="utf-8"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_parquet", required=True)
    ap.add_argument("--clusters_tsv", required=True)
    ap.add_argument("--codebook_csv", required=True)
    ap.add_argument("--outdir", default="analysis/final/triplet_stats")
    ap.add_argument("--sections", default=SECTIONS_DEFAULT)
    ap.add_argument("--role", default="user", choices=["user","assistant","all"])
    ap.add_argument("--include_complements", action="store_true")
    ap.add_argument("--write_occurrences", action="store_true")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--per_transcript_topk", type=int, default=10)
    args = ap.parse_args()

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    cb = load_codebook(Path(args.codebook_csv))
    cl = load_clusters(Path(args.clusters_tsv))
    tok = load_tokens(Path(args.token_parquet), sections=sections, role=args.role)

    if tok.empty:
        raise SystemExit(f"No tokens loaded. Check --sections and --role. Sections passed: {sections}")

    occ = extract_occurrences(tok, cl, include_complements=args.include_complements)
    if occ.empty:
        raise SystemExit("No triplet occurrences extracted. Check token fields (sent_id/head/deprel).")

    # join labels
    key = ["subj_lemma","pred_lemma","obj_lemma","action_lemma","pattern","voice"]
    occ_lab = occ.merge(cb[key+["agent_binary"]], on=key, how="left")
    occ_lab["agent_binary"] = occ_lab["agent_binary"].fillna("").astype(str)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.write_occurrences:
        occ_lab.to_csv(outdir / "triplet_occurrences_labeled.csv", index=False)

    write_final(outdir, occ_lab, topk=args.topk, per_transcript_topk=args.per_transcript_topk)
    print("Done. Final pack written to:", outdir.resolve())

if __name__ == "__main__":
    main()