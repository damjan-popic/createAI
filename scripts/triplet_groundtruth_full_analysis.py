#!/usr/bin/env python3
from __future__ import annotations

"""
CreateAI v2 — Triplet Ground Truth Full Analysis (occurrence-level, section-aware)

This script treats your *triplet codebook candidates CSV* as SOURCE OF TRUTH for labels,
then re-extracts triplet occurrences from stanza_out/token_level.parquet (section+role scoped),
joins occurrences -> ground-truth labels via normalized lemma-triplet signature, and produces
comprehensive stats + plots answering:

  - Where is AI vs HUMAN overall?
  - Where (sections) does AI dominate?
  - Where (clusters) does AI dominate?
  - Section × cluster AI-share heatmaps
  - Which predicates/objects/actions characterize AI vs HUMAN by section
  - Complement use (xcomp/ccomp) by label/section
  - Passive/agent patterns by label/section
  - AI-in-context/oblique diagnostics by label/section
  - Transcript-level AI share (overall and by section)

Inputs
------
1) Ground truth codebook CSV (source of truth):
   Must include at least:
     agent_binary, subj_lemma, pred_lemma, obj_lemma, action_lemma
   (can include other fields; ignored)

2) Token parquet:
   stanza_out/token_level.parquet
   Requires:
     transcript_id, role, section, sent_id, word_id, text, lemma, upos, head, deprel

3) Cluster mapping:
   interviews_clustered.tsv
   Requires:
     transcript_id (plus group/long_cluster if available)

Recommended run:
  python scripts/triplet_groundtruth_full_analysis.py \
    --codebook_csv "analysis/triplet_codebook/triplets_codebook_candidates.csv" \
    --token_parquet stanza_out/token_level.parquet \
    --clusters_tsv interviews_clustered.tsv \
    --outdir analysis/triplet_groundtruth_analysis \
    --sections job_description,dynamic,walkthrough,project_example \
    --role user \
    --include_complements

"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pyarrow.dataset as ds  # optional for faster parquet filtering
except Exception:
    ds = None


# -------------------------
# Constants / helpers
# -------------------------

DEFAULT_SECTIONS = ["job_description", "dynamic", "walkthrough", "project_example"]

SUBJ_ACTIVE = {"nsubj", "csubj", "nsubj:outer", "csubj:outer"}
SUBJ_PASS = {"nsubj:pass", "csubj:pass", "nsubj:pass:outer", "csubj:pass:outer"}
AGENT_PASS = {"obl:agent", "agent"}
OBJ_DEPS = {"obj", "iobj"}
COMP_DEPS = {"xcomp", "ccomp"}

# Phrase span expansion for subj/obj/action heads
SPAN_DEPS = {"det", "amod", "compound", "nummod", "nmod:poss", "case", "fixed", "flat", "name", "neg"}
DETERMINERS = {"the", "a", "an"}

# AI markers for sentence-level flags
AI_TEXT_RE = re.compile(
    r"""(?ix)
    \bchatgpt\b
    |\bgpt(?:\s*[-–]?\s*\d+)?\b
    |\bllm\b
    |\bai\b
    |\bopenai\b
    |\banthropic\b
    |\bclaude\b
    |\bcopilot\b
    |\bgemini\b
    |\bbard\b
    |\bmidjourney\b
    |\bdall[- ]?e\b
    |\bstable\s*diffusion\b
    """
)

OBJ_PLACEHOLDERS = {
    "it", "this", "that", "these", "those",
    "something", "anything", "everything", "nothing",
    "someone", "somebody", "anyone", "anybody",
    "which", "who", "whom", "what"
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def norm_lower(x) -> str:
    return norm(x).lower()


def norm_label(x) -> str:
    s = norm(x).upper()
    return s if s in {"AI", "HUMAN"} else ""


def mk_sig_norm(subj: str, pred: str, obj: str, action: str) -> str:
    # strict machine signature for joining
    return "||".join([norm_lower(subj), norm_lower(pred), norm_lower(obj), norm_lower(action)])


def mk_sig_pretty(subj: str, pred: str, obj: str, action: str) -> str:
    subj = norm(subj)
    pred = norm(pred)
    obj = norm(obj)
    action = norm(action)
    if action:
        return f"{subj} :: {pred} :: {obj} :: {action}"
    return f"{subj} :: {pred} :: {obj}"


def safe_int(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


# -------------------------
# Loaders
# -------------------------

def load_codebook(codebook_csv: Path) -> pd.DataFrame:
    cb = pd.read_csv(codebook_csv)
    need = ["agent_binary", "subj_lemma", "pred_lemma", "obj_lemma", "action_lemma"]
    missing = [c for c in need if c not in cb.columns]
    if missing:
        raise ValueError(f"Codebook missing columns {missing}. Found: {list(cb.columns)}")

    cb = cb.copy()
    cb["agent_binary"] = cb["agent_binary"].apply(norm_label)
    for c in ["subj_lemma", "pred_lemma", "obj_lemma", "action_lemma"]:
        cb[c] = cb[c].fillna("")

    cb["sig_norm"] = cb.apply(
        lambda r: mk_sig_norm(r["subj_lemma"], r["pred_lemma"], r["obj_lemma"], r["action_lemma"]), axis=1
    )

    # de-dupe by signature (source of truth = first row)
    cb = cb.drop_duplicates(subset=["sig_norm"]).copy()
    return cb


def load_clusters(clusters_tsv: Path) -> pd.DataFrame:
    cl = pd.read_csv(clusters_tsv, sep="\t", dtype=str)
    if "transcript_id" not in cl.columns:
        raise ValueError(f"{clusters_tsv} must contain transcript_id. Found: {list(cl.columns)}")
    if "group" not in cl.columns:
        cl["group"] = ""
    if "long_cluster" not in cl.columns:
        cl["long_cluster"] = ""
    return cl[["transcript_id", "group", "long_cluster"]].fillna("")


def load_tokens(token_parquet: Path, sections: List[str], role: str) -> pd.DataFrame:
    if ds is not None:
        dataset = ds.dataset(str(token_parquet), format="parquet")
        filt = ds.field("section").isin(sections)
        if role != "all":
            filt = filt & (ds.field("role") == role)
        table = dataset.to_table(filter=filt)
        df = table.to_pandas()
    else:
        df = pd.read_parquet(token_parquet)
        df = df[df["section"].isin(sections)]
        if role != "all":
            df = df[df["role"] == role]

    need = ["transcript_id", "role", "section", "sent_id", "word_id", "text", "lemma", "upos", "head", "deprel"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"token_level.parquet missing {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["lemma"] = df["lemma"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["upos"] = df["upos"].astype(str).str.strip().str.upper()
    df["deprel"] = df["deprel"].astype(str).str.strip()
    df["section"] = df["section"].astype(str).str.strip()
    df["role"] = df["role"].astype(str).str.strip().str.lower()

    df["word_id"] = pd.to_numeric(df["word_id"], errors="coerce").astype("Int64")
    df["head"] = pd.to_numeric(df["head"], errors="coerce").astype("Int64")
    df["sent_id"] = pd.to_numeric(df["sent_id"], errors="coerce").astype("Int64")

    return df.dropna(subset=["transcript_id", "section", "sent_id", "word_id", "deprel"])


# -------------------------
# Sentence diagnostics (AI presence)
# -------------------------

def sentence_ai_flags(sent: pd.DataFrame) -> Tuple[bool, bool, int]:
    """
    Returns (ai_in_ctx, ai_in_obl, ai_token_hits).
    - ai_in_ctx: any AI marker in sentence text
    - ai_in_obl: any AI marker in an oblique (deprel startswith 'obl')
    - ai_token_hits: number of tokens matching AI markers (text or lemma)
    """
    texts = " ".join(sent["text"].astype(str).tolist())
    ai_in_ctx = bool(AI_TEXT_RE.search(texts))

    hit = sent["text"].astype(str).str.contains(AI_TEXT_RE, regex=True) | sent["lemma"].astype(str).str.contains(AI_TEXT_RE, regex=True)
    ai_token_hits = int(hit.sum())

    obl_mask = sent["deprel"].astype(str).str.startswith("obl")
    if obl_mask.any():
        obl_ai = (
            sent.loc[obl_mask, "text"].astype(str).str.contains(AI_TEXT_RE, regex=True) |
            sent.loc[obl_mask, "lemma"].astype(str).str.contains(AI_TEXT_RE, regex=True)
        ).any()
    else:
        obl_ai = False

    return ai_in_ctx, bool(obl_ai), ai_token_hits


# -------------------------
# Dependency helpers
# -------------------------

def build_children_map(sent_df: pd.DataFrame) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {}
    for r in sent_df.itertuples(index=False):
        if pd.isna(r.head) or pd.isna(r.word_id):
            continue
        h = int(r.head)
        wid = int(r.word_id)
        if h <= 0:
            continue
        children.setdefault(h, []).append(wid)
    return children


def token_by_id(sent_df: pd.DataFrame) -> Dict[int, dict]:
    d = {}
    for r in sent_df.itertuples(index=False):
        if pd.isna(r.word_id):
            continue
        d[int(r.word_id)] = {
            "word_id": int(r.word_id),
            "text": r.text,
            "lemma": r.lemma,
            "upos": r.upos,
            "deprel": r.deprel,
            "head": int(r.head) if pd.notna(r.head) else 0,
        }
    return d


def pick_dependents(children: Dict[int, List[int]], tok: Dict[int, dict], head_id: int, rels: set) -> List[int]:
    out: List[int] = []
    for cid in children.get(head_id, []):
        dep = tok.get(cid)
        if dep and dep["deprel"] in rels:
            out.append(cid)
    return out


def span_for_head(head_id: int, tok: Dict[int, dict], children: Dict[int, List[int]], max_len: int = 8) -> Tuple[str, str]:
    ids = [head_id]
    for cid in children.get(head_id, []):
        dep = tok.get(cid)
        if dep and dep["deprel"] in SPAN_DEPS:
            ids.append(cid)

    ids = sorted(set(ids))[:max_len]
    texts = [tok[i]["text"] for i in ids if i in tok]
    lems = [tok[i]["lemma"] for i in ids if i in tok]
    return " ".join(map(str, texts)).strip(), " ".join(map(str, lems)).strip()


def repair_determiner_object(obj_id: int, sent_tok: Dict[int, dict]) -> int:
    t = sent_tok.get(obj_id)
    if not t:
        return obj_id
    l = norm_lower(t.get("lemma"))
    tx = norm_lower(t.get("text"))
    if t.get("upos") == "DET" or l in DETERMINERS or tx in DETERMINERS:
        nxt = obj_id + 1
        if nxt in sent_tok:
            return nxt
    return obj_id


# -------------------------
# Triplet extraction (occurrence-level)
# -------------------------

def extract_triplets_occurrences(tokens: pd.DataFrame,
                                clusters: Optional[pd.DataFrame],
                                include_complements: bool,
                                max_subj: int = 3,
                                max_obj: int = 3,
                                max_comp: int = 2) -> pd.DataFrame:
    """
    Returns one row per triplet occurrence, with metadata:
      transcript_id, section, sent_id, role,
      group, long_cluster,
      voice, pattern,
      subj_role, obj_role,
      subj_lemma, pred_lemma, obj_lemma, action_lemma,
      subj_text/pred_text/obj_text/action_text,
      ai_in_ctx, ai_in_obl, ai_token_hits
    """
    df = tokens.copy()

    if clusters is not None:
        df = df.merge(clusters, on="transcript_id", how="left")
    else:
        df["group"] = ""
        df["long_cluster"] = ""

    df = df.dropna(subset=["transcript_id", "sent_id", "word_id", "deprel", "lemma", "text", "section"])
    df["role"] = df["role"].astype(str).str.lower()

    out_rows: List[dict] = []
    group_cols = ["transcript_id", "sent_id", "section", "role", "group", "long_cluster"]

    for key, sent in df.groupby(group_cols, sort=False):
        transcript_id, sent_id, section, role, group, long_cluster = key
        sent = sent.sort_values("word_id")

        ai_in_ctx, ai_in_obl, ai_token_hits = sentence_ai_flags(sent)

        children = build_children_map(sent)
        tok = token_by_id(sent)

        for v in sent.itertuples(index=False):
            if v.upos not in ("VERB", "AUX"):
                continue

            v_id = int(v.word_id)
            v_lemma = v.lemma
            v_text = v.text

            passive_agents = pick_dependents(children, tok, v_id, AGENT_PASS)
            active_subjs = pick_dependents(children, tok, v_id, SUBJ_ACTIVE)
            passive_subjs = pick_dependents(children, tok, v_id, SUBJ_PASS)
            objects = pick_dependents(children, tok, v_id, OBJ_DEPS)

            comps: List[int] = []
            if include_complements:
                comp_ids = pick_dependents(children, tok, v_id, COMP_DEPS)
                # keep only VERB/AUX complements
                for cid in comp_ids:
                    dep = tok.get(cid)
                    if dep and dep.get("upos") in {"VERB", "AUX"}:
                        comps.append(cid)

            # choose subject with priority
            voice = "ACT"
            subj_role = "SUBJECT"
            subj_ids: List[int] = []

            if passive_agents:
                voice = "PASS"
                subj_role = "PASS_AGENT"
                subj_ids = passive_agents[:max_subj]
            elif active_subjs:
                subj_ids = active_subjs[:max_subj]
            elif passive_subjs:
                voice = "PASS"
                subj_role = "PATIENT_SUBJ"
                subj_ids = passive_subjs[:max_subj]
            else:
                continue  # no subject

            # object selection
            obj_ids = objects[:max_obj]
            obj_role = "OBJECT"
            if not obj_ids and passive_subjs:
                obj_ids = passive_subjs[:max_obj]
                obj_role = "PASS_PATIENT"

            comp_ids = comps[:max_comp]

            for s_id in subj_ids:
                subj_head = tok.get(s_id, {})
                subj_lemma = subj_head.get("lemma", "")
                subj_span_text, _ = span_for_head(s_id, tok, children)
                subj_text = subj_span_text or subj_head.get("text", "")

                # If no obj and no comp => SV
                if not obj_ids and not comp_ids:
                    out_rows.append({
                        "transcript_id": transcript_id,
                        "sent_id": int(sent_id),
                        "section": section,
                        "role": role,
                        "group": group,
                        "long_cluster": long_cluster,
                        "voice": voice,
                        "pattern": "SV",
                        "subj_role": subj_role,
                        "subj_lemma": subj_lemma,
                        "subj_text": subj_text,
                        "pred_lemma": v_lemma,
                        "pred_text": v_text,
                        "obj_role": "",
                        "obj_lemma": "",
                        "obj_text": "",
                        "action_lemma": "",
                        "action_text": "",
                        "ai_in_ctx": ai_in_ctx,
                        "ai_in_obl": ai_in_obl,
                        "ai_token_hits": ai_token_hits,
                    })
                    continue

                for o_id in (obj_ids or [None]):
                    obj_lemma = obj_text = ""
                    obj_role_eff = obj_role

                    if o_id is not None:
                        o_id2 = repair_determiner_object(int(o_id), tok)
                        obj_head = tok.get(o_id2, {})
                        obj_lemma = obj_head.get("lemma", "")
                        obj_span_text, _ = span_for_head(o_id2, tok, children)
                        obj_text = obj_span_text or obj_head.get("text", "")

                    if comp_ids:
                        for c_id in comp_ids:
                            c = tok.get(c_id, {})
                            c_lemma = c.get("lemma", "")
                            c_span_text, _ = span_for_head(c_id, tok, children)
                            c_text = c_span_text or c.get("text", "")

                            out_rows.append({
                                "transcript_id": transcript_id,
                                "sent_id": int(sent_id),
                                "section": section,
                                "role": role,
                                "group": group,
                                "long_cluster": long_cluster,
                                "voice": voice,
                                "pattern": "SVOxCOMP" if obj_lemma else "SVxCOMP",
                                "subj_role": subj_role,
                                "subj_lemma": subj_lemma,
                                "subj_text": subj_text,
                                "pred_lemma": v_lemma,
                                "pred_text": v_text,
                                "obj_role": obj_role_eff if obj_lemma else "",
                                "obj_lemma": obj_lemma,
                                "obj_text": obj_text,
                                "action_lemma": c_lemma,
                                "action_text": c_text,
                                "ai_in_ctx": ai_in_ctx,
                                "ai_in_obl": ai_in_obl,
                                "ai_token_hits": ai_token_hits,
                            })
                    else:
                        out_rows.append({
                            "transcript_id": transcript_id,
                            "sent_id": int(sent_id),
                            "section": section,
                            "role": role,
                            "group": group,
                            "long_cluster": long_cluster,
                            "voice": voice,
                            "pattern": "SVO" if obj_lemma else "SV",
                            "subj_role": subj_role,
                            "subj_lemma": subj_lemma,
                            "subj_text": subj_text,
                            "pred_lemma": v_lemma,
                            "pred_text": v_text,
                            "obj_role": obj_role_eff if obj_lemma else "",
                            "obj_lemma": obj_lemma,
                            "obj_text": obj_text,
                            "action_lemma": "",
                            "action_text": "",
                            "ai_in_ctx": ai_in_ctx,
                            "ai_in_obl": ai_in_obl,
                            "ai_token_hits": ai_token_hits,
                        })

    return pd.DataFrame(out_rows)


# -------------------------
# Analysis helpers
# -------------------------

def shares_table(df: pd.DataFrame, group_cols: List[str], label_col: str = "label") -> pd.DataFrame:
    """
    Correct implementation: count rows per (group_cols + label), then compute totals from that.
    Returns: group_cols + label + count + total + share
    """
    g = (
        df.groupby(group_cols + [label_col], as_index=False)
          .size()
          .rename(columns={"size": "count"})
    )
    tot = (
        g.groupby(group_cols, as_index=False)["count"]
          .sum()
          .rename(columns={"count": "total"})
    )
    out = g.merge(tot, on=group_cols, how="left")
    out["share"] = out["count"] / out["total"].replace({0: np.nan})
    return out


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path, topn: int = 30, rotate: int = 45) -> None:
    if df.empty:
        return
    d = df.sort_values(y, ascending=False).head(topn).copy()
    plt.figure(figsize=(12, 6))
    plt.bar(d[x].astype(str), d[y].astype(float).values)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_stacked_share(df: pd.DataFrame, index_col: str, class_col: str, share_col: str, title: str, outpath: Path, topn: int = 30) -> None:
    if df.empty:
        return
    piv = df.pivot_table(index=index_col, columns=class_col, values=share_col, aggfunc="sum", fill_value=0)
    if len(piv) > topn:
        piv = piv.iloc[:topn]
    ax = piv.plot(kind="bar", stacked=True, figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel("Share")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.figure.savefig(outpath, dpi=220)
    plt.close(ax.figure)


def heatmap(pivot: pd.DataFrame, title: str, outpath: Path, cbar_label: str) -> None:
    if pivot.empty:
        return
    plt.figure(figsize=(12, 7))
    data = pivot.values.astype(float)
    plt.imshow(data, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.colorbar(label=cbar_label)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def ai_share_from_counts(ai: pd.Series, human: pd.Series) -> pd.Series:
    denom = (ai.fillna(0) + human.fillna(0)).replace({0: np.nan})
    return ai.fillna(0) / denom


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook_csv", default="analysis/triplet_codebook/triplets_codebook_candidates.csv",
                    help="Ground-truth triplet codebook (source of truth labels).")
    ap.add_argument("--token_parquet", default="stanza_out/token_level.parquet")
    ap.add_argument("--clusters_tsv", default="interviews_clustered.tsv")
    ap.add_argument("--outdir", default="analysis/triplet_groundtruth_analysis")
    ap.add_argument("--sections", default=",".join(DEFAULT_SECTIONS),
                    help="Comma-separated list of sections to include.")
    ap.add_argument("--role", default="user", choices=["user", "assistant", "all"])
    ap.add_argument("--include_complements", action="store_true",
                    help="Include xcomp/ccomp actions (recommended).")
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--min_heatmap_n", type=int, default=10, help="Min labeled occurrences for a (section, cluster) cell.")
    ap.add_argument("--heatmap_top_clusters", type=int, default=12, help="How many clusters to show in heatmap columns.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(figdir)

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    # --- Load source-of-truth codebook
    cb = load_codebook(Path(args.codebook_csv))
    cb = cb[cb["agent_binary"].isin(["AI", "HUMAN"])].copy()  # ignore blanks as labels by design
    label_map = cb[["sig_norm", "agent_binary"]].rename(columns={"agent_binary": "label"})

    # --- Load clusters + tokens
    cl = load_clusters(Path(args.clusters_tsv))
    tok = load_tokens(Path(args.token_parquet), sections=sections, role=args.role)

    # --- Extract occurrences
    occ = extract_triplets_occurrences(tok, cl, include_complements=args.include_complements)
    if occ.empty:
        raise SystemExit("No triplet occurrences extracted. Check --sections/--role and parquet content.")

    # --- Signatures and join to labels
    occ["sig_norm"] = occ.apply(lambda r: mk_sig_norm(r["subj_lemma"], r["pred_lemma"], r["obj_lemma"], r["action_lemma"]), axis=1)
    occ["triplet_sig"] = occ.apply(lambda r: mk_sig_pretty(r["subj_lemma"], r["pred_lemma"], r["obj_lemma"], r["action_lemma"]), axis=1)
    occ = occ.merge(label_map, on="sig_norm", how="left")
    occ["label"] = occ["label"].fillna("").apply(norm_label)

    # --- Derived metrics
    occ["has_action"] = occ["action_lemma"].fillna("").astype(str).str.strip().ne("")
    occ["obj_is_placeholder"] = occ["obj_lemma"].fillna("").astype(str).str.lower().isin(OBJ_PLACEHOLDERS)
    occ["is_passive"] = occ["voice"].fillna("").astype(str).str.upper().eq("PASS")

    # labeled only subset
    occ_lab = occ[occ["label"].isin(["AI", "HUMAN"])].copy()

    # Save full occurrence-level table
    occ.to_csv(outdir / "occurrences_labeled.csv", index=False)

    # --- Overview
    overview = pd.DataFrame([{
        "sections": ",".join(sections),
        "role": args.role,
        "include_complements": bool(args.include_complements),
        "occ_total": int(len(occ)),
        "occ_labeled": int(len(occ_lab)),
        "unique_types_total": int(occ["sig_norm"].nunique()),
        "unique_types_labeled": int(occ_lab["sig_norm"].nunique()),
        "unique_transcripts": int(occ["transcript_id"].nunique()),
        "unique_clusters": int(occ["group"].nunique()),
    }])
    overview.to_csv(outdir / "summary_overview.csv", index=False)

    # -------------------------
    # Core: overall counts
    # -------------------------
    label_counts = occ.groupby("label").size().reset_index(name="occurrence_count")
    label_counts["label"] = label_counts["label"].replace({"": "BLANK"})
    label_counts.to_csv(outdir / "label_counts_overall.csv", index=False)
    plot_bar(label_counts, "label", "occurrence_count",
             "Triplet occurrences by label (incl. BLANK)", figdir / "label_counts_overall.png",
             topn=10, rotate=0)

    # -------------------------
    # Where is AI vs HUMAN: section shares
    # -------------------------
    sec_counts = occ_lab.groupby(["section", "label"]).size().reset_index(name="count")
    sec_counts.to_csv(outdir / "label_by_section_counts.csv", index=False)

    sec_shares = shares_table(occ_lab, ["section"], "label")
    sec_shares.to_csv(outdir / "label_by_section_shares.csv", index=False)

    plot_stacked_share(sec_shares, "section", "label", "share",
                       "AI vs HUMAN share by section (labeled occurrences only)",
                       figdir / "label_share_by_section.png", topn=50)

    # AI share per section (single series)
    sec_piv = sec_counts.pivot_table(index="section", columns="label", values="count", aggfunc="sum", fill_value=0)
    sec_ai_share = ai_share_from_counts(sec_piv.get("AI", pd.Series(dtype=float)), sec_piv.get("HUMAN", pd.Series(dtype=float)))
    sec_ai_share = sec_ai_share.reset_index().rename(columns={0: "AI_share"})
    sec_ai_share.to_csv(outdir / "ai_share_by_section.csv", index=False)
    plot_bar(sec_ai_share, "section", "AI_share",
             "AI share by section (AI / (AI + HUMAN))", figdir / "ai_share_by_section.png", topn=50, rotate=45)

    # -------------------------
    # Where is AI vs HUMAN: cluster shares (group)
    # -------------------------
    cl_counts = occ_lab.groupby(["group", "label"]).size().reset_index(name="count")
    cl_counts.to_csv(outdir / "label_by_cluster_counts.csv", index=False)

    cl_shares = shares_table(occ_lab, ["group"], "label")
    cl_shares.to_csv(outdir / "label_by_cluster_shares.csv", index=False)

    # top clusters plot
    cl_tot = occ_lab.groupby("group").size().sort_values(ascending=False).reset_index(name="total")
    cl_top = cl_tot.head(args.topn)["group"].tolist()
    cl_shares_top = cl_shares[cl_shares["group"].isin(cl_top)].copy()

    plot_stacked_share(cl_shares_top, "group", "label", "share",
                       "AI vs HUMAN share by cluster (top clusters)",
                       figdir / "label_share_by_cluster_top.png", topn=args.topn)

    # -------------------------
    # Section × cluster heatmap (AI share)
    # -------------------------
    sec_group_counts = occ_lab.groupby(["section", "group", "label"]).size().reset_index(name="count")
    # pivot for ai share per cell
    cell = sec_group_counts.pivot_table(index=["section", "group"], columns="label", values="count", aggfunc="sum", fill_value=0).reset_index()
    cell["TOTAL"] = cell.get("AI", 0) + cell.get("HUMAN", 0)
    cell = cell[cell["TOTAL"] >= args.min_heatmap_n].copy()
    cell["AI_share"] = cell.get("AI", 0) / cell["TOTAL"].replace({0: np.nan})

    # choose top clusters (overall volume) to show as heatmap columns
    heat_groups = occ_lab.groupby("group").size().sort_values(ascending=False).head(args.heatmap_top_clusters).index.tolist()
    heat = cell[cell["group"].isin(heat_groups)].pivot_table(index="section", columns="group", values="AI_share", aggfunc="mean")

    heat.to_csv(outdir / "ai_share_heatmap_section_x_cluster.csv")
    heatmap(heat.fillna(0.0),
            "AI share by section × cluster (top clusters; filtered by min counts)",
            figdir / "ai_share_heatmap_section_x_cluster.png",
            cbar_label="AI share (AI / (AI + HUMAN))")

    # -------------------------
    # Predicates / objects / actions: overall + by section
    # -------------------------
    # Predicates overall
    pred_overall = occ_lab.groupby(["label", "pred_lemma"]).size().reset_index(name="count") \
        .sort_values(["label", "count"], ascending=[True, False])
    pred_overall.to_csv(outdir / "top_predicates_by_label_overall.csv", index=False)

    for lab in ["AI", "HUMAN"]:
        d = pred_overall[pred_overall["label"] == lab].head(args.topn)
        plot_bar(d, "pred_lemma", "count", f"Top predicates — {lab}", figdir / f"top_predicates_{lab}.png", topn=args.topn, rotate=65)

    # Predicates by section
    pred_sec = occ_lab.groupby(["section", "label", "pred_lemma"]).size().reset_index(name="count") \
        .sort_values(["section", "label", "count"], ascending=[True, True, False])
    pred_sec.to_csv(outdir / "top_predicates_by_label_section.csv", index=False)

    # Objects overall (exclude placeholders)
    occ_lab_no_ph = occ_lab[~occ_lab["obj_is_placeholder"]].copy()
    obj_overall = occ_lab_no_ph.groupby(["label", "obj_lemma"]).size().reset_index(name="count") \
        .sort_values(["label", "count"], ascending=[True, False])
    obj_overall.to_csv(outdir / "top_objects_by_label_overall.csv", index=False)

    for lab in ["AI", "HUMAN"]:
        d = obj_overall[obj_overall["label"] == lab].head(args.topn)
        plot_bar(d, "obj_lemma", "count", f"Top objects (non-placeholder) — {lab}", figdir / f"top_objects_{lab}.png", topn=args.topn, rotate=65)

    # Actions overall (complements only)
    act_overall = occ_lab[occ_lab["has_action"]].groupby(["label", "action_lemma"]).size().reset_index(name="count") \
        .sort_values(["label", "count"], ascending=[True, False])
    act_overall.to_csv(outdir / "top_actions_by_label_overall.csv", index=False)

    for lab in ["AI", "HUMAN"]:
        d = act_overall[act_overall["label"] == lab].head(args.topn)
        plot_bar(d, "action_lemma", "count", f"Top complement actions — {lab}", figdir / f"top_actions_{lab}.png", topn=args.topn, rotate=65)

    # -------------------------
    # Triplet TYPES: overall + by section (by occurrence counts)
    # -------------------------
    type_overall = occ_lab.groupby(["label", "triplet_sig"]).size().reset_index(name="count") \
        .sort_values(["label", "count"], ascending=[True, False])
    type_overall.to_csv(outdir / "top_triplets_by_label_overall.csv", index=False)

    for lab in ["AI", "HUMAN"]:
        d = type_overall[type_overall["label"] == lab].head(args.topn)
        plot_bar(d, "triplet_sig", "count", f"Top triplet types — {lab}", figdir / f"top_triplet_types_{lab}.png", topn=15, rotate=65)

    type_sec = occ_lab.groupby(["section", "label", "triplet_sig"]).size().reset_index(name="count") \
        .sort_values(["section", "label", "count"], ascending=[True, True, False])
    type_sec.to_csv(outdir / "top_triplets_by_label_section.csv", index=False)

    # -------------------------
    # Complements (xcomp/ccomp) rate by section × label
    # -------------------------
    comp_rate = occ_lab.groupby(["section", "label"]).agg(
        occurrences=("sig_norm", "size"),
        with_action=("has_action", "sum")
    ).reset_index()
    comp_rate["action_rate"] = comp_rate["with_action"] / comp_rate["occurrences"].replace({0: np.nan})
    comp_rate.to_csv(outdir / "complement_rate_by_label_section.csv", index=False)

    comp_piv = comp_rate.pivot_table(index="section", columns="label", values="action_rate", aggfunc="mean").fillna(0)
    comp_piv.to_csv(outdir / "complement_rate_pivot.csv")
    heatmap(comp_piv, "Complement-action rate by section × label", figdir / "complement_action_rate_heatmap.png", "Rate")

    # -------------------------
    # Passive voice patterns by section × label
    # -------------------------
    passive = occ_lab.groupby(["section", "label"]).agg(
        occurrences=("sig_norm", "size"),
        passive=("is_passive", "sum")
    ).reset_index()
    passive["passive_rate"] = passive["passive"] / passive["occurrences"].replace({0: np.nan})
    passive.to_csv(outdir / "passive_rate_by_label_section.csv", index=False)

    pass_piv = passive.pivot_table(index="section", columns="label", values="passive_rate", aggfunc="mean").fillna(0)
    pass_piv.to_csv(outdir / "passive_rate_pivot.csv")
    heatmap(pass_piv, "Passive rate by section × label", figdir / "passive_rate_heatmap.png", "Rate")

    # -------------------------
    # AI context diagnostics (ai_in_ctx / ai_in_obl / token hits)
    # -------------------------
    ctx = occ_lab.groupby(["section", "label"]).agg(
        occurrences=("sig_norm", "size"),
        ai_in_ctx=("ai_in_ctx", "sum"),
        ai_in_obl=("ai_in_obl", "sum"),
        ai_token_hits=("ai_token_hits", "sum"),
        placeholder_obj=("obj_is_placeholder", "sum")
    ).reset_index()

    ctx["ai_in_ctx_rate"] = ctx["ai_in_ctx"] / ctx["occurrences"].replace({0: np.nan})
    ctx["ai_in_obl_rate"] = ctx["ai_in_obl"] / ctx["occurrences"].replace({0: np.nan})
    ctx["placeholder_obj_rate"] = ctx["placeholder_obj"] / ctx["occurrences"].replace({0: np.nan})
    ctx["ai_token_hits_per_occ"] = ctx["ai_token_hits"] / ctx["occurrences"].replace({0: np.nan})
    ctx.to_csv(outdir / "ai_context_rates_by_label_section.csv", index=False)

    # Heatmaps for these rates
    ctx_piv1 = ctx.pivot_table(index="section", columns="label", values="ai_in_ctx_rate", aggfunc="mean").fillna(0)
    ctx_piv2 = ctx.pivot_table(index="section", columns="label", values="ai_in_obl_rate", aggfunc="mean").fillna(0)
    ctx_piv3 = ctx.pivot_table(index="section", columns="label", values="placeholder_obj_rate", aggfunc="mean").fillna(0)

    ctx_piv1.to_csv(outdir / "ai_in_ctx_rate_pivot.csv")
    ctx_piv2.to_csv(outdir / "ai_in_obl_rate_pivot.csv")
    ctx_piv3.to_csv(outdir / "placeholder_obj_rate_pivot.csv")

    heatmap(ctx_piv1, "AI-in-context rate by section × label", figdir / "ai_in_ctx_rate_heatmap.png", "Rate")
    heatmap(ctx_piv2, "AI-in-oblique rate by section × label", figdir / "ai_in_obl_rate_heatmap.png", "Rate")
    heatmap(ctx_piv3, "Placeholder-object rate by section × label", figdir / "placeholder_obj_rate_heatmap.png", "Rate")

    # -------------------------
    # Transcript-level AI share (overall + by section)
    # -------------------------
    tr = occ_lab.groupby(["transcript_id", "label"]).size().reset_index(name="count")
    tr_p = tr.pivot_table(index="transcript_id", columns="label", values="count", aggfunc="sum", fill_value=0)
    tr_p["TOTAL"] = tr_p.get("AI", 0) + tr_p.get("HUMAN", 0)
    tr_p["AI_share"] = tr_p.get("AI", 0) / tr_p["TOTAL"].replace({0: np.nan})
    tr_p.reset_index().sort_values("AI_share", ascending=False).to_csv(outdir / "transcript_ai_share.csv", index=False)

    tr_sec = occ_lab.groupby(["transcript_id", "section", "label"]).size().reset_index(name="count")
    tr_sec_p = tr_sec.pivot_table(index=["transcript_id", "section"], columns="label", values="count", aggfunc="sum", fill_value=0)
    tr_sec_p["TOTAL"] = tr_sec_p.get("AI", 0) + tr_sec_p.get("HUMAN", 0)
    tr_sec_p["AI_share"] = tr_sec_p.get("AI", 0) / tr_sec_p["TOTAL"].replace({0: np.nan})
    tr_sec_p.reset_index().to_csv(outdir / "transcript_section_ai_share.csv", index=False)

    # -------------------------
    # Print pointers
    # -------------------------
    print("\nOK — wrote outputs to:", outdir)
    print("Figures in:", figdir)
    print("\nStart here:")
    print(" - label_by_section_shares.csv  + figures/label_share_by_section.png")
    print(" - ai_share_by_section.csv      + figures/ai_share_by_section.png")
    print(" - label_by_cluster_shares.csv  + figures/label_share_by_cluster_top.png")
    print(" - ai_share_heatmap_section_x_cluster.csv + figures/ai_share_heatmap_section_x_cluster.png")
    print(" - top_predicates_by_label_section.csv")
    print(" - ai_context_rates_by_label_section.csv")
    print(" - transcript_ai_share.csv")


if __name__ == "__main__":
    main()