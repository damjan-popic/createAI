#!/usr/bin/env python3
"""
Dependency-based Agency Analysis (codebook-projected)

Inputs:
- stanza_out/token_level.parquet
- analysis/targets_with_categories.xlsx (sheet=codebook; lemma | pos_group | CATEGORY)
- interviews_clustered.tsv (transcript_id | group | long_cluster)

Outputs:
- analysis/categorized/edges/verb_dep_edges.{parquet,csv}
- analysis/categorized/agg/subject_share_by_category.csv
- analysis/categorized/agg/subject_share_by_category_cluster.csv
- analysis/categorized/sanity/verb_sanity_checks.csv

Core logic:
- Build dependency edges via self-join on (transcript_id, section, sent_id) + head word_id
- Keep only edges where HEAD is a codebook VERB
- Extract dependents for: subjects (nsubj/csubj + passive variants), objects (obj/iobj), agents (obl:agent/agent)
- Classify subject dependents into HUMAN | AI | OTHER using simple lemma/text heuristics
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


SUBJECT_DEPREL_PREFIXES = ("nsubj", "csubj")  # includes nsubj:pass, csubj:pass, etc.
OBJECT_DEPRELS = {"obj", "iobj"}
AGENT_DEPREL_PREFIXES = ("obl:agent", "agent")  # includes obl:agent, agent


def norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def casefold(x) -> str:
    return norm_str(x).strip().casefold()


def deprel_is_subject(deprel: str) -> bool:
    d = casefold(deprel)
    return any(d == p or d.startswith(p + ":") for p in SUBJECT_DEPREL_PREFIXES)


def deprel_is_object(deprel: str) -> bool:
    return casefold(deprel) in OBJECT_DEPRELS


def deprel_is_agent(deprel: str) -> bool:
    d = casefold(deprel)
    return any(d == p or d.startswith(p + ":") for p in AGENT_DEPREL_PREFIXES)


def dep_role_from_deprel(deprel: str) -> Optional[str]:
    if deprel_is_subject(deprel):
        return "SUBJECT"
    if deprel_is_object(deprel):
        return "OBJECT"
    if deprel_is_agent(deprel):
        return "AGENT"
    return None


@dataclass(frozen=True)
class Heuristics:
    human_lemmas: Set[str]
    human_texts: Set[str]
    ai_lemmas: Set[str]
    ai_text_patterns: List[re.Pattern]


def default_heuristics() -> Heuristics:
    # Keep these small and editable; this is corpus-linguistics style, not ontology.
    human_lemmas = {
        # EN pronouns
        "i", "we", "me", "us", "my", "our", "mine", "ours", "myself", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs",
        # SL pronouns (minimal)
        "jaz", "ti", "on", "ona", "ono", "mi", "vi", "oni", "one",
        "mene", "meni", "me", "nama", "nam", "tebe", "tebi", "te", "vam",
        "moj", "moja", "moje", "naš", "naša", "naše",
        # IT pronouns (minimal)
        "io", "noi", "mi", "mio", "mia", "miei", "mie", "tu", "voi", "lui", "lei", "loro",
    }

    # Sometimes lemma is unhelpful (e.g., "I" lemmatized to "I" vs "I"); text helps.
    human_texts = {  # casefolded tokens
        "i", "we", "me", "us", "my", "our", "you", "he", "she", "they",
        "jaz", "mi", "ti", "vi", "on", "ona",
        "io", "noi", "tu", "voi", "lui", "lei",
    }

    ai_lemmas = {
        "ai", "a.i.", "artificial_intelligence",
        "model", "llm", "gpt", "chatgpt", "assistant", "system", "bot",
        "machine", "algorithm", "tool", "computer", "software",
        "copilot", "claude", "gemini", "bard", "openai",
    }

    # Patterns catch text variants like "GPT-4", "GPT4", "ChatGPT", "AI", "A.I."
    ai_text_patterns = [
        re.compile(r"\bai\b", flags=re.IGNORECASE),
        re.compile(r"\ba\.i\.\b", flags=re.IGNORECASE),
        re.compile(r"\bgpt[- ]?\d+\b", flags=re.IGNORECASE),
        re.compile(r"\bchatgpt\b", flags=re.IGNORECASE),
        re.compile(r"\bllm\b", flags=re.IGNORECASE),
        re.compile(r"\bcopilot\b", flags=re.IGNORECASE),
        re.compile(r"\bclaude\b", flags=re.IGNORECASE),
        re.compile(r"\bgemini\b", flags=re.IGNORECASE),
        re.compile(r"\bbard\b", flags=re.IGNORECASE),
        re.compile(r"\bopenai\b", flags=re.IGNORECASE),
    ]

    return Heuristics(
        human_lemmas={casefold(x) for x in human_lemmas},
        human_texts={casefold(x) for x in human_texts},
        ai_lemmas={casefold(x) for x in ai_lemmas},
        ai_text_patterns=ai_text_patterns,
    )


def classify_subject_agent(dep_lemma: str, dep_text: str, heur: Heuristics) -> str:
    l = casefold(dep_lemma)
    t = casefold(dep_text)

    if l in heur.human_lemmas or t in heur.human_texts:
        return "HUMAN"

    if l in heur.ai_lemmas:
        return "AI"

    # pattern match on surface form
    for pat in heur.ai_text_patterns:
        if pat.search(dep_text or ""):
            return "AI"

    return "OTHER"


def read_codebook(codebook_xlsx: str, sheet: str = "codebook") -> pd.DataFrame:
    cb = pd.read_excel(codebook_xlsx, sheet_name=sheet, engine="openpyxl")

    # --- robust column normalization (case/whitespace-insensitive) ---
    cols = list(cb.columns)
    cols_cf = {c: str(c).strip().casefold() for c in cols}

    def pick_col(*wanted_cf: str) -> str:
        wanted = {w.casefold() for w in wanted_cf}
        for c in cols:
            if cols_cf[c] in wanted:
                return c
        raise ValueError(f"Codebook missing one of columns {sorted(wanted)}. Found: {cols}")

    lemma_col = pick_col("lemma")
    pos_col = pick_col("pos_group", "pos group", "pos")
    cat_col = pick_col("category", "CATEGORY")

    cb = cb.rename(columns={lemma_col: "lemma", pos_col: "pos_group", cat_col: "CATEGORY"})

    # --- canonical processing ---
    cb = cb.copy()
    cb["lemma_cf"] = cb["lemma"].astype(str).str.strip().str.casefold()
    cb["pos_group_cf"] = cb["pos_group"].astype(str).str.strip().str.casefold()
    cb["CATEGORY"] = cb["CATEGORY"].astype(str).str.strip()

    cb["pos_norm"] = cb["pos_group_cf"].replace(
        {
            "v": "verb",
            "verb": "verb",
            "verbs": "verb",
            "n": "noun",
            "noun": "noun",
            "nouns": "noun",
        }
    )

    return cb[["lemma", "lemma_cf", "pos_norm", "CATEGORY"]]


def build_dependency_edges(tokens: pd.DataFrame) -> pd.DataFrame:
    """
    Self-join tokens to attach head token data to each dependent token.
    Join key assumes head is word_id within the same sentence (typical Stanza/CoNLL-U).
    """
    t = tokens.copy()

    # enforce types
    for col in ["word_id", "head", "sent_id"]:
        t[col] = pd.to_numeric(t[col], errors="coerce").astype("Int64")

    # drop roots and broken heads
    t = t[t["head"].notna() & (t["head"] > 0)]

    head_cols = {
        "word_id": "head_word_id",
        "text": "head_text",
        "lemma": "head_lemma",
        "upos": "head_upos",
        "xpos": "head_xpos",
        "feats": "head_feats",
        "deprel": "head_deprel",  # usually irrelevant, but traceable
    }

    heads = t[["transcript_id", "section", "sent_id", "word_id", "text", "lemma", "upos", "xpos", "feats", "deprel"]].rename(
        columns=head_cols
    )

    deps = t.rename(
        columns={
            "word_id": "dep_word_id",
            "text": "dep_text",
            "lemma": "dep_lemma",
            "upos": "dep_upos",
            "xpos": "dep_xpos",
            "feats": "dep_feats",
            "deprel": "dep_deprel",
        }
    )

    # Join dependents to their heads
    edges = deps.merge(
        heads,
        how="left",
        left_on=["transcript_id", "section", "sent_id", "head"],
        right_on=["transcript_id", "section", "sent_id", "head_word_id"],
        validate="m:1",
    )

    # Some heads may not join if IDs are inconsistent; keep but mark as missing
    return edges


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_parquet", default="stanza_out/token_level.parquet")
    ap.add_argument("--codebook_xlsx", default="analysis/targets_with_categories.xlsx")
    ap.add_argument("--codebook_sheet", default="codebook")
    ap.add_argument("--clusters_tsv", default="interviews_clustered.tsv")
    ap.add_argument("--out_dir", default="analysis/categorized")
    ap.add_argument("--restrict_dependents_to_codebook", action="store_true",
                    help="If set, keep only edges where dependent lemma is ALSO in the 400-lemma codebook.")
    ap.add_argument("--sanity_verbs", default="take,use,control",
                    help="Comma-separated verb lemmas for sanity checks (lemmas, not surface).")
    args = ap.parse_args()

    heur = default_heuristics()

    cb = read_codebook(args.codebook_xlsx, sheet=args.codebook_sheet)
    cb_verbs = cb[cb["pos_norm"] == "verb"].copy()
    cb_nouns = cb[cb["pos_norm"] == "noun"].copy()

    verb_set = set(cb_verbs["lemma_cf"].tolist())
    noun_set = set(cb_nouns["lemma_cf"].tolist())

    # map lemma_cf -> category (for verbs, later for nouns too)
    verb_cat: Dict[str, str] = dict(zip(cb_verbs["lemma_cf"], cb_verbs["CATEGORY"]))
    noun_cat: Dict[str, str] = dict(zip(cb_nouns["lemma_cf"], cb_nouns["CATEGORY"]))

    tokens = pd.read_parquet(args.tokens_parquet)
    required_cols = {
        "transcript_id", "role", "section", "section_raw", "sent_id", "word_id",
        "text", "lemma", "upos", "xpos", "feats", "head", "deprel"
    }
    missing = required_cols - set(tokens.columns)
    if missing:
        raise ValueError(f"token_level.parquet missing columns: {missing}")

    # Attach cluster labels
    clusters = pd.read_csv(args.clusters_tsv, sep="\t")
    if not {"transcript_id", "group", "long_cluster"} <= set(clusters.columns):
        raise ValueError("interviews_clustered.tsv must have transcript_id, group, long_cluster")

    tokens = tokens.merge(clusters[["transcript_id", "group", "long_cluster"]],
                          on="transcript_id", how="left")

    # Build edges
    edges = build_dependency_edges(tokens)

    # Identify dep_role
    edges["dep_role"] = edges["dep_deprel"].apply(dep_role_from_deprel)
    edges = edges[edges["dep_role"].notna()].copy()

    # Add casefolded lemmas
    edges["head_lemma_cf"] = edges["head_lemma"].astype(str).str.strip().str.casefold()
    edges["dep_lemma_cf"] = edges["dep_lemma"].astype(str).str.strip().str.casefold()

    # Keep only edges where HEAD is a codebook verb
    edges = edges[edges["head_lemma_cf"].isin(verb_set)].copy()

    # Optionally restrict dependents to codebook too
    if args.restrict_dependents_to_codebook:
        codebook_all = verb_set | noun_set
        edges = edges[edges["dep_lemma_cf"].isin(codebook_all)].copy()

    # Verb category
    edges["verb_lemma"] = edges["head_lemma"]
    edges["verb_category"] = edges["head_lemma_cf"].map(verb_cat).fillna("")

    # Dep fields
    edges["dep_text"] = edges["dep_text"].astype(str)
    edges["dep_lemma"] = edges["dep_lemma"].astype(str)

    # Cluster fields
    edges["cluster"] = edges["group"].fillna("")
    edges["cluster_long"] = edges["long_cluster"].fillna("")

    # Speaker role (user/assistant)
    edges["speaker_role"] = edges["role"].fillna("")

    # Agent attribution (subjects only)
    edges["agent_type"] = ""
    subj_mask = edges["dep_role"] == "SUBJECT"
    edges.loc[subj_mask, "agent_type"] = edges.loc[subj_mask].apply(
        lambda r: classify_subject_agent(r["dep_lemma"], r["dep_text"], heur),
        axis=1
    )

    # Output selection (human-readable but still traceable)
    out_edges = edges[[
        "transcript_id",
        "speaker_role",
        "section",
        "sent_id",
        "head_word_id",
        "dep_word_id",
        "verb_lemma",
        "verb_category",
        "dep_role",
        "dep_deprel",
        "dep_text",
        "dep_lemma",
        "agent_type",
        "cluster",
        "cluster_long",
    ]].rename(columns={"head_word_id": "verb_word_id", "dep_deprel": "deprel"})

    # Write outputs
    edges_dir = os.path.join(args.out_dir, "edges")
    agg_dir = os.path.join(args.out_dir, "agg")
    sanity_dir = os.path.join(args.out_dir, "sanity")
    ensure_dir(edges_dir)
    ensure_dir(agg_dir)
    ensure_dir(sanity_dir)

    out_parq = os.path.join(edges_dir, "verb_dep_edges.parquet")
    out_csv = os.path.join(edges_dir, "verb_dep_edges.csv")
    out_edges.to_parquet(out_parq, index=False)
    out_edges.to_csv(out_csv, index=False, encoding="utf-8")

    # Aggregates: subject share by category × agent_type
    subj = out_edges[out_edges["dep_role"] == "SUBJECT"].copy()
    subj["agent_type"] = subj["agent_type"].replace("", "OTHER")

    agg_cat = (
        subj.groupby(["verb_category", "agent_type"])
        .size()
        .reset_index(name="count")
    )
    totals = agg_cat.groupby("verb_category")["count"].transform("sum")
    agg_cat["share"] = agg_cat["count"] / totals

    agg_cat_out = os.path.join(agg_dir, "subject_share_by_category.csv")
    agg_cat.to_csv(agg_cat_out, index=False, encoding="utf-8")

    # Aggregates: category × cluster × agent_type
    agg_cat_cluster = (
        subj.groupby(["verb_category", "cluster", "agent_type"])
        .size()
        .reset_index(name="count")
    )
    totals2 = agg_cat_cluster.groupby(["verb_category", "cluster"])["count"].transform("sum")
    agg_cat_cluster["share"] = agg_cat_cluster["count"] / totals2

    agg_cat_cluster_out = os.path.join(agg_dir, "subject_share_by_category_cluster.csv")
    agg_cat_cluster.to_csv(agg_cat_cluster_out, index=False, encoding="utf-8")

    # Sanity checks for representative verbs
    sanity_verbs = [casefold(v) for v in args.sanity_verbs.split(",") if v.strip()]
    sanity_rows = []
    for vcf in sanity_verbs:
        v_edges = out_edges[out_edges["verb_lemma"].astype(str).str.strip().str.casefold() == vcf]
        if v_edges.empty:
            sanity_rows.append({"verb": vcf, "note": "NO EDGES FOUND", "dep_role": "", "count": 0, "top_dep_lemmas": ""})
            continue

        # count by dep_role
        for dep_role, g in v_edges.groupby("dep_role"):
            top_deps = (
                g["dep_lemma"].astype(str).str.strip().str.casefold()
                .value_counts()
                .head(10)
                .to_dict()
            )
            sanity_rows.append({
                "verb": vcf,
                "note": "",
                "dep_role": dep_role,
                "count": int(len(g)),
                "top_dep_lemmas": "; ".join([f"{k}:{v}" for k, v in top_deps.items()]),
            })

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_out = os.path.join(sanity_dir, "verb_sanity_checks.csv")
    sanity_df.to_csv(sanity_out, index=False, encoding="utf-8")

    print("OK ✅")
    print(f"Wrote edges:   {out_parq}")
    print(f"Wrote edges:   {out_csv}")
    print(f"Wrote agg:     {agg_cat_out}")
    print(f"Wrote agg:     {agg_cat_cluster_out}")
    print(f"Wrote sanity:  {sanity_out}")


if __name__ == "__main__":
    main()
