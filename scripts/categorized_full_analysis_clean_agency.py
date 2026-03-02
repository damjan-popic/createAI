#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CreateAI — Clean agency aggregation
#
# Key methodological construct:
# - CATEGORY labels WHAT type of creative act is being discussed
#   (AI_USE / CONTROL / AUTHORSHIP / IDEATION).
# - AGENT_CLASS labels WHO is the grammatical agent for those acts
#   (HUMAN / AI / OTHER), based on subject/agent dependents.
#
# Agency vs patient:
# - Active subjects (nsubj/csubj) and explicit passive agents (obl:agent/agent)
#   are treated as AGENTS.
# - Passive subjects (nsubj:pass/csubj:pass) and objects (obj/iobj)
#   are treated as PATIENTS (not agents).
#
# Optional add-on:
# - For xcomp/ccomp complements, derive "embedded action" events:
#   controller verb (e.g. use/let/help) + xcomp action verb (e.g. generate/write).
#   Agent is taken from the controller's agent (subject/obl:agent).
# ============================================================


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_str(s) -> str:
    return str(s).strip()

def norm_lower(s) -> str:
    return norm_str(s).lower()

def normalize_pos_group(s: str) -> str:
    s = norm_upper(s)
    if s in {"V", "VERB"}:
        return "VERB"
    if s in {"N", "NOUN"}:
        return "NOUN"
    return s

def norm_upper(s) -> str:
    return norm_str(s).upper()

def find_codebook_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.lower().strip(): c for c in df.columns}
    # allow common variants
    lemma = cols.get("lemma")
    pos = cols.get("pos_group") or cols.get("pos") or cols.get("posgroup")
    cat = cols.get("category") or cols.get("CATEGORY") or cols.get("cat")
    if not lemma or not pos or not cat:
        raise ValueError(f"Codebook must contain columns lemma, pos_group, CATEGORY. Found: {list(df.columns)}")
    return lemma, pos, cat

def detect_cluster_col(df: pd.DataFrame) -> str:
    # prefer short label if present
    if "group" in df.columns:
        return "group"
    if "long_cluster" in df.columns:
        return "long_cluster"
    return "group"


# -------------------------
# Agent classification (simple, adjustable)
# -------------------------

DEFAULT_AI_LEMMAS = {
    # Keep strict & explicit. Do NOT include generic "tool/system/assistant" by default.
    "ai", "llm", "model", "chatgpt", "gpt", "openai", "anthropic", "claude",
    "copilot", "gemini", "bard", "midjourney", "dalle", "stable-diffusion", "stablediffusion",
}

AI_AMBIGUOUS = {"assistant", "tool", "system", "software", "machine", "chatbot", "bot"}

AI_TEXT_RE = re.compile(
    r"""(?ix)
    \bchatgpt\b
    |\bgpt(?:\s*[-–]?\s*\d+)?\b
    |\bllm\b
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

DEFAULT_HUMAN_LEMMAS = {
    # Pronouns + common forms
    "i", "we", "you", "he", "she", "they",
    "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "their",
    "mine", "ours", "yours", "hers", "theirs",

    # Person nouns (extend as needed)
    "person", "people", "human", "user", "client", "customer",
    "writer", "author", "artist", "designer", "editor", "producer",
    "team", "colleague", "coworker", "manager", "boss", "student", "teacher",
}

HUMAN_PRONOUNS = {"i", "we", "you", "he", "she", "they"}

NON_AGENT_DEFAULTS = {"it", "this", "that", "there", "something", "anything", "everything", "nothing"}

def classify_entity(lemma: str, text: str, ai_set: set, human_set: set) -> str:
    """
    Returns: HUMAN | AI | OTHER

    IMPORTANT: classification is based on the dependent token (subject/agent/patient),
    not on speaker metadata.
    """
    l = norm_lower(lemma)
    t = norm_lower(text)

    if l in NON_AGENT_DEFAULTS:
        return "OTHER"

    # HUMAN first (pronouns and explicit human nouns)
    if l in HUMAN_PRONOUNS or l in human_set or t in human_set:
        return "HUMAN"

    # AI (explicit lemma) or strong text pattern (GPT-4, ChatGPT, etc.)
    if l in ai_set:
        return "AI"
    if AI_TEXT_RE.search(t):
        return "AI"

    # Ambiguous AI-ish nouns default to OTHER unless you override ai_set via --ai_lex
    if l in AI_AMBIGUOUS:
        return "OTHER"

    return "OTHER"


# -------------------------
# I/O
# -------------------------

def load_codebook(codebook_xlsx: Path, sheet: str) -> pd.DataFrame:
    cb_raw = pd.read_excel(codebook_xlsx, sheet_name=sheet)
    lemma_col, pos_col, cat_col = find_codebook_cols(cb_raw)

    cb = cb_raw[[lemma_col, pos_col, cat_col]].copy()
    cb.columns = ["lemma", "pos_group", "category"]
    # Avoid turning NaN into literal "nan" strings.
    cb["lemma"] = cb["lemma"].fillna("").astype(str).str.strip().str.lower()
    cb["pos_group"] = cb["pos_group"].fillna("").astype(str).apply(normalize_pos_group)
    cb["category"] = cb["category"].fillna("").astype(str).str.strip().str.upper()

    # keep only valid labeled rows
    valid_cats = {"AI_USE", "CONTROL", "AUTHORSHIP", "IDEATION"}
    cb = cb[cb["lemma"].ne("") & cb["category"].isin(valid_cats) & cb["pos_group"].isin({"VERB", "NOUN"})]
    cb = cb.drop_duplicates(subset=["lemma"])
    return cb

def load_clusters(clusters_tsv: Path) -> pd.DataFrame:
    cl = pd.read_csv(clusters_tsv, sep="\t")
    if "transcript_id" not in cl.columns:
        raise ValueError(f"{clusters_tsv} must contain transcript_id. Found: {list(cl.columns)}")

    keep = ["transcript_id"]
    if "group" in cl.columns:
        keep.append("group")
    if "long_cluster" in cl.columns:
        keep.append("long_cluster")
    cl = cl[keep].copy()
    if len(keep) == 1:
        cl["group"] = "UNKNOWN"
    return cl

def load_tokens(token_parquet: Path, role: str) -> pd.DataFrame:
    tok = pd.read_parquet(token_parquet)
    need = ["transcript_id", "section", "sent_id", "word_id", "text", "lemma", "upos", "head", "deprel"]
    missing = [c for c in need if c not in tok.columns]
    if missing:
        raise ValueError(f"token_level.parquet missing columns {missing}. Found: {list(tok.columns)}")

    tok = tok.copy()
    tok["lemma"] = tok["lemma"].astype(str).str.strip().str.lower()
    tok["text"] = tok["text"].astype(str).str.strip()
    tok["upos"] = tok["upos"].astype(str).str.strip().str.upper()
    tok["deprel"] = tok["deprel"].astype(str).str.strip()
    tok["section"] = tok["section"].astype(str).str.strip()

    if "role" in tok.columns and role != "all":
        tok["role"] = tok["role"].astype(str).str.strip().str.lower()
        tok = tok[tok["role"] == role]

    tok["word_id"] = pd.to_numeric(tok["word_id"], errors="coerce").fillna(-1).astype(int)
    tok["head"] = pd.to_numeric(tok["head"], errors="coerce").fillna(-1).astype(int)
    return tok


# -------------------------
# Token-share tables (lexical presence, codebook-only)
# -------------------------

def restrict_to_codebook(tok: pd.DataFrame, cb: pd.DataFrame, clusters: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    df = tok.merge(cb, on="lemma", how="inner")
    df = df.merge(clusters, on="transcript_id", how="left")
    cluster_col = detect_cluster_col(df)
    if cluster_col not in df.columns:
        df["group"] = "UNKNOWN"
        cluster_col = "group"
    df[cluster_col] = df[cluster_col].fillna("UNKNOWN")

    # count tokens
    df["token_count"] = 1
    return df, cluster_col

def category_token_share(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby([group_col, "category"], as_index=False)["token_count"].sum()
    tot = df.groupby([group_col], as_index=False)["token_count"].sum().rename(columns={"token_count": "group_total"})
    g = g.merge(tot, on=group_col, how="left")
    g["share"] = g["token_count"] / g["group_total"]
    return g.sort_values([group_col, "share"], ascending=[True, False])


# -------------------------
# Dependency extraction
# -------------------------

ACTIVE_SUBJ_DEPRELS = {"nsubj", "csubj", "nsubj:outer", "csubj:outer"}
PASSIVE_SUBJ_DEPRELS = {"nsubj:pass", "csubj:pass", "nsubj:pass:outer", "csubj:pass:outer"}
OBJ_DEPRELS = {"obj", "iobj"}
AGENT_DEPRELS = {"obl:agent", "agent"}
COMP_DEPRELS = {"xcomp", "ccomp"}

def extract_edges(tok: pd.DataFrame, cb: pd.DataFrame, clusters: pd.DataFrame, include_complements: bool) -> Tuple[pd.DataFrame, str]:
    """
    Heads: categorized CODEBOOK VERBS only.
    Dependents: unrestricted token universe, filtered to relevant deprels.
    """
    cb_verbs = cb[cb["pos_group"].eq("VERB")].copy()
    heads = tok.merge(cb_verbs, on="lemma", how="inner")
    # Stanza may tag auxiliaries as AUX; treat them as eligible verb heads.
    heads = heads[heads["upos"].isin({"VERB", "AUX"})].copy()

    keys = ["transcript_id", "section", "sent_id"]
    heads = heads[keys + ["word_id", "lemma", "category"]].rename(
        columns={"word_id": "verb_word_id", "lemma": "verb_lemma", "category": "verb_category"}
    )

    wanted = set(ACTIVE_SUBJ_DEPRELS) | set(PASSIVE_SUBJ_DEPRELS) | set(OBJ_DEPRELS) | set(AGENT_DEPRELS)
    if include_complements:
        wanted |= set(COMP_DEPRELS)

    deps = tok[keys + ["word_id", "head", "deprel", "lemma", "text", "upos"]].copy()
    deps = deps[deps["deprel"].isin(wanted)].copy()
    deps = deps.rename(columns={"word_id": "dep_word_id", "lemma": "dep_lemma", "text": "dep_text", "upos": "dep_upos"})

    merged = deps.merge(
        heads,
        left_on=keys + ["head"],
        right_on=keys + ["verb_word_id"],
        how="inner",
    )

    def edge_role_detail(r: str) -> str:
        rr = r.strip()
        if rr in ACTIVE_SUBJ_DEPRELS:
            return "SUBJECT"
        if rr in PASSIVE_SUBJ_DEPRELS:
            return "SUBJECT_PASS"
        if rr in AGENT_DEPRELS:
            return "AGENT"
        if rr in OBJ_DEPRELS:
            return "OBJECT"
        if rr in COMP_DEPRELS:
            return rr.upper()
        return "OTHER"

    def agency_role(r: str) -> str:
        rr = r.strip()
        if rr in ACTIVE_SUBJ_DEPRELS or rr in AGENT_DEPRELS:
            return "AGENT"
        if rr in PASSIVE_SUBJ_DEPRELS or rr in OBJ_DEPRELS:
            return "PATIENT"
        return "OTHER"

    merged["edge_role_detail"] = merged["deprel"].apply(edge_role_detail)
    merged["agency_role"] = merged["deprel"].apply(agency_role)

    # attach cluster label
    merged = merged.merge(clusters, on="transcript_id", how="left")
    cluster_col = detect_cluster_col(merged)
    if cluster_col not in merged.columns:
        merged["group"] = "UNKNOWN"
        cluster_col = "group"
    merged[cluster_col] = merged[cluster_col].fillna("UNKNOWN")

    return merged, cluster_col


# -------------------------
# Clean aggregation logic
# -------------------------

def _shares(df: pd.DataFrame, group_cols: List[str], class_col: str, count_col: str = "edge_count") -> pd.DataFrame:
    """
    Compute counts + within-group shares for class_col.
    """
    g = df.groupby(group_cols + [class_col], as_index=False)[count_col].sum()
    tot = df.groupby(group_cols, as_index=False)[count_col].sum().rename(columns={count_col: "total"})
    out = g.merge(tot, on=group_cols, how="left")
    out["share"] = out[count_col] / out["total"]
    return out

def compute_agency_tables(edges: pd.DataFrame, cluster_col: str, ai_set: set, human_set: set) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of tables:
      - agent_shares_by_category
      - agent_shares_by_category_section
      - agent_shares_by_category_cluster
      - agent_shares_by_category_section_cluster
      - top_agents_by_category_full
      - patient_shares_by_category (optional interpretive)
      - passive_subject_rate_by_category
    """
    out: Dict[str, pd.DataFrame] = {}
    edges = edges.copy()
    edges["edge_count"] = 1

    # -----------------
    # AGENTS: active subjects + explicit passive agents
    # -----------------
    agent_edges = edges[edges["agency_role"].eq("AGENT")].copy()
    agent_edges["agent_class"] = agent_edges.apply(
        lambda r: classify_entity(r["dep_lemma"], r["dep_text"], ai_set, human_set),
        axis=1
    )

    # Shares within specific groupings (category is the *type of act*, agent_class is *who drives it*)
    out["agent_shares_by_category"] = _shares(agent_edges, ["verb_category"], "agent_class")
    out["agent_shares_by_category_section"] = _shares(agent_edges, ["verb_category", "section"], "agent_class")
    out["agent_shares_by_category_cluster"] = _shares(agent_edges, ["verb_category", cluster_col], "agent_class").rename(columns={cluster_col: "cluster"})
    out["agent_shares_by_category_section_cluster"] = _shares(agent_edges, ["verb_category", "section", cluster_col], "agent_class").rename(columns={cluster_col: "cluster"})

    # Pooled across categories (useful for the "dynamic is human-driven" hypothesis check)
    out["agent_shares_by_section"] = _shares(agent_edges, ["section"], "agent_class")
    out["agent_shares_by_cluster"] = _shares(agent_edges, [cluster_col], "agent_class").rename(columns={cluster_col: "cluster"})
    out["agent_shares_by_section_cluster"] = _shares(agent_edges, ["section", cluster_col], "agent_class").rename(columns={cluster_col: "cluster"})
    # Top agent lemmas by category/class
    top_agents = (
        agent_edges.groupby(["verb_category", "agent_class", "dep_lemma"], as_index=False)["edge_count"].sum()
        .sort_values(["verb_category", "agent_class", "edge_count"], ascending=[True, True, False])
        .rename(columns={"dep_lemma": "agent_lemma"})
    )
    out["top_agents_by_category_full"] = top_agents

    # -----------------
    # PATIENTS: objects + passive subjects
    # Useful to interpret "AI as object/instrument" vs "AI as agent".
    # -----------------
    patient_edges = edges[edges["agency_role"].eq("PATIENT")].copy()
    patient_edges["entity_class"] = patient_edges.apply(
        lambda r: classify_entity(r["dep_lemma"], r["dep_text"], ai_set, human_set),
        axis=1
    )
    out["patient_shares_by_category"] = _shares(patient_edges, ["verb_category"], "entity_class").rename(columns={"entity_class": "patient_class"})
    out["patient_shares_by_category_section"] = _shares(patient_edges, ["verb_category", "section"], "entity_class").rename(columns={"entity_class": "patient_class"})

    # Passive subject rate (how often category verbs appear in passive constructions)
    subj_any = edges[edges["edge_role_detail"].isin(["SUBJECT", "SUBJECT_PASS"])].copy()
    if len(subj_any):
        pass_ct = subj_any[subj_any["edge_role_detail"].eq("SUBJECT_PASS")].groupby(["verb_category"], as_index=False)["edge_count"].sum().rename(columns={"edge_count": "passive_subject_edges"})
        all_ct = subj_any.groupby(["verb_category"], as_index=False)["edge_count"].sum().rename(columns={"edge_count": "all_subject_edges"})
        rate = pass_ct.merge(all_ct, on="verb_category", how="outer").fillna(0)
        rate["passive_subject_rate"] = rate["passive_subject_edges"] / rate["all_subject_edges"].replace({0: pd.NA})
        out["passive_subject_rate_by_category"] = rate.sort_values("verb_category")
    else:
        out["passive_subject_rate_by_category"] = pd.DataFrame(columns=["verb_category", "passive_subject_edges", "all_subject_edges", "passive_subject_rate"])

    # sort outputs for readability
    for k, v in out.items():
        if "verb_category" in v.columns:
            out[k] = v.sort_values([c for c in ["verb_category", "section", "cluster", "agent_class", "patient_class", "share", "edge_count"] if c in v.columns],
                                   ascending=True)

    return out


def compute_embedded_action_tables(edges: pd.DataFrame, cluster_col: str, cb: pd.DataFrame,
                                  ai_set: set, human_set: set) -> Dict[str, pd.DataFrame]:
    """
    Derive embedded action events for xcomp/ccomp:
    controller verb (categorized) -> complement verb (if categorized in codebook)
    agent = controller agent (active subject or explicit passive agent).
    """
    out: Dict[str, pd.DataFrame] = {}

    # action category map (verbs only)
    cbv = cb[cb["pos_group"].eq("VERB")].copy()
    action_cat = dict(zip(cbv["lemma"], cbv["category"]))

    keys = ["transcript_id", "section", "sent_id", "verb_word_id"]

    # agent assignments per controller head (can be multiple in coordination; keep all)
    agent_edges = edges[edges["agency_role"].eq("AGENT")].copy()
    agent_edges["agent_class"] = agent_edges.apply(
        lambda r: classify_entity(r["dep_lemma"], r["dep_text"], ai_set, human_set),
        axis=1
    )
    head_agents = agent_edges[keys + ["verb_lemma", "verb_category", cluster_col, "agent_class", "dep_lemma", "dep_text"]].rename(
        columns={cluster_col: "cluster", "dep_lemma": "agent_lemma", "dep_text": "agent_text"}
    )

    # complement edges
    comp = edges[edges["edge_role_detail"].isin(["XCOMP", "CCOMP"])].copy()
    comp = comp[comp["dep_upos"].isin({"VERB", "AUX"})].copy()

    # merge to get controller agent(s)
    comp = comp.merge(head_agents, on=keys + ["verb_lemma", "verb_category"], how="left")

    # derive action
    comp["action_lemma"] = comp["dep_lemma"]
    comp["action_category"] = comp["action_lemma"].map(action_cat)

    # keep only where action is categorized (codebook verb with category)
    comp = comp[comp["action_category"].notna()].copy()
    comp["edge_count"] = 1
    comp["agent_class"] = comp["agent_class"].fillna("OTHER")

    # shares of agent_class by action_category (optionally by section/cluster)
    out["embedded_action_agent_shares_by_action_category"] = _shares(comp, ["action_category"], "agent_class")
    out["embedded_action_agent_shares_by_action_category_section"] = _shares(comp, ["action_category", "section"], "agent_class")
    out["embedded_action_agent_shares_by_action_category_cluster"] = _shares(comp, ["action_category", "cluster"], "agent_class")

    # controller_category -> action_category matrix (counts)
    mat = comp.groupby(["verb_category", "action_category", "agent_class"], as_index=False)["edge_count"].sum()
    out["controller_to_action_matrix"] = mat.sort_values(["verb_category", "action_category", "edge_count"], ascending=[True, True, False])

    # top action lemmas per action_category
    top_actions = (
        comp.groupby(["action_category", "action_lemma"], as_index=False)["edge_count"].sum()
        .sort_values(["action_category", "edge_count"], ascending=[True, False])
    )
    out["top_action_lemmas_by_action_category"] = top_actions

    return out


# -------------------------
# Plotting (minimal)
# -------------------------

def plot_stacked(df: pd.DataFrame, index_cols: List[str], class_col: str, share_col: str, title: str, outpath: Path) -> None:
    if df.empty:
        return
    idx = index_cols[0] if len(index_cols) == 1 else tuple(index_cols)
    piv = df.pivot_table(index=index_cols, columns=class_col, values=share_col, aggfunc="sum", fill_value=0)
    plt.figure(figsize=(12, 6))
    piv.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.ylabel("Share")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_parquet", default="stanza_out/token_level.parquet")
    ap.add_argument("--codebook_xlsx", default="analysis/targets_with_categories.xlsx")
    ap.add_argument("--codebook_sheet", default="codebook")
    ap.add_argument("--clusters_tsv", default="interviews_clustered.tsv")
    ap.add_argument("--outdir", default="analysis/categorized_clean")
    ap.add_argument("--role", default="all", choices=["all", "user", "assistant"])
    ap.add_argument("--ai_lex", default="", help="Optional newline lexicon file to add AI markers (lemmas).")
    ap.add_argument("--human_lex", default="", help="Optional newline lexicon file to add HUMAN markers (lemmas).")
    ap.add_argument("--include_complements", action="store_true", help="Include xcomp/ccomp edges and derive embedded actions.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(figdir)

    cb = load_codebook(Path(args.codebook_xlsx), args.codebook_sheet)
    cl = load_clusters(Path(args.clusters_tsv))
    tok = load_tokens(Path(args.token_parquet), role=args.role)

    # Token-share tables (codebook-only presence)
    df_codebook, cluster_col = restrict_to_codebook(tok, cb, cl)
    by_section = category_token_share(df_codebook, "section")
    by_section.to_csv(outdir / "category_token_share_by_section.csv", index=False)
    by_cluster = category_token_share(df_codebook, cluster_col)
    by_cluster.to_csv(outdir / "category_token_share_by_cluster.csv", index=False)

    # Dependency edges
    edges, edge_cluster_col = extract_edges(tok, cb, cl, include_complements=args.include_complements)
    edges.to_csv(outdir / "verb_dependency_edges_full.csv", index=False)

    # Build lexicons
    ai_set = set(DEFAULT_AI_LEMMAS)
    human_set = set(DEFAULT_HUMAN_LEMMAS)
    if args.ai_lex:
        ai_set |= {norm_lower(x) for x in Path(args.ai_lex).read_text(encoding="utf-8").splitlines() if x.strip()}
    if args.human_lex:
        human_set |= {norm_lower(x) for x in Path(args.human_lex).read_text(encoding="utf-8").splitlines() if x.strip()}

    # Agency tables
    agency = compute_agency_tables(edges, edge_cluster_col, ai_set, human_set)
    for name, table in agency.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    # Simple figures: agent shares by category + by section
    plot_stacked(
        agency["agent_shares_by_category"],
        ["verb_category"], "agent_class", "share",
        "AGENT share by verb category (active subjects + explicit passive agents)",
        figdir / "agent_shares_by_category.png"
    )

    plot_stacked(
        agency["agent_shares_by_section"],
        ["section"], "agent_class", "share",
        "AGENT share by section (within each section, pooled across categories)",
        figdir / "agent_shares_by_section.png"
    )

    # Embedded actions (optional)
    if args.include_complements:
        embedded = compute_embedded_action_tables(edges, edge_cluster_col, cb, ai_set, human_set)
        for name, table in embedded.items():
            table.to_csv(outdir / f"{name}.csv", index=False)

        plot_stacked(
            embedded["embedded_action_agent_shares_by_action_category"],
            ["action_category"], "agent_class", "share",
            "Embedded action agent share (xcomp/ccomp actions by category)",
            figdir / "embedded_action_agent_shares_by_action_category.png"
        )

    print("\nWrote outputs to:", outdir)
    print("Key outputs (clean constructs):")
    print(" - category_token_share_by_section.csv (lexical presence; NOT agency)")
    print(" - agent_shares_by_category*.csv (AGENT role only; passive subjects excluded)")
    print(" - patient_shares_by_category*.csv (OBJECT + passive subjects; interpretive)")
    if args.include_complements:
        print(" - embedded_action_agent_shares_by_action_category*.csv (xcomp/ccomp derived actions)")
        print(" - controller_to_action_matrix.csv")

if __name__ == "__main__":
    main()
