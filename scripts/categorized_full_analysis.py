#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_str(s) -> str:
    return str(s).strip()

def norm_lower(s) -> str:
    return str(s).strip().lower()

def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else float("-inf")

def detect_cluster_col(df: pd.DataFrame) -> str:
    if "group" in df.columns:
        return "group"
    if "long_cluster" in df.columns:
        return "long_cluster"
    return "group"

def find_codebook_cols(cb: pd.DataFrame) -> Tuple[str, str, str]:
    cols = {c.strip().lower(): c for c in cb.columns}
    lemma_col = cols.get("lemma")
    pos_col = cols.get("pos_group") or cols.get("posgroup") or cols.get("pos")
    cat_col = cols.get("category") or cols.get("categories") or cols.get("cat")
    if not lemma_col or not pos_col or not cat_col:
        raise ValueError(f"Codebook must have lemma, pos_group, CATEGORY. Found: {list(cb.columns)}")
    return lemma_col, pos_col, cat_col

def normalize_pos_group(pos: str) -> str:
    p = norm_str(pos).upper()
    if "VERB" in p:
        return "VERB"
    if "NOUN" in p or "PROPN" in p:
        return "NOUN"
    # Keep others if they exist, but your codebook should be VERB/NOUN
    return p

def is_verb_upos(upos: str) -> bool:
    return norm_str(upos).upper() == "VERB"

def is_nounish_upos(upos: str) -> bool:
    u = norm_str(upos).upper()
    return u in {"NOUN", "PROPN"}


# -------------------------
# Agent classification (simple, adjustable)
# -------------------------

DEFAULT_AI_LEMMAS = {
    "ai", "claude", "anthropic", "model", "llm", "gpt", "chatgpt",
    "assistant", "chatbot", "system", "tool", "bot", "machine"
}

DEFAULT_HUMAN_LEMMAS = {
    "i", "we", "me", "us", "my", "our", "mine", "ours",
    "you", "your", "yours",
    "person", "people", "human", "user", "client", "customer",
    "writer", "author", "artist", "designer", "editor", "producer",
    "team", "colleague", "coworker", "manager", "boss"
}

HUMAN_PRONOUNS = {"i", "we", "me", "us", "my", "our", "mine", "ours"}

def classify_agent(lemma: str, text: str, ai_set: set, human_set: set) -> str:
    """
    Returns: HUMAN | AI | OTHER
    Priority: explicit AI markers > explicit human markers > OTHER
    """
    l = norm_lower(lemma)
    t = norm_lower(text)

    if l in ai_set or t in ai_set:
        return "AI"
    if l in human_set or t in human_set or l in HUMAN_PRONOUNS:
        return "HUMAN"
    # crude heuristic: "it" etc -> OTHER
    return "OTHER"


# -------------------------
# Core computations
# -------------------------

def load_codebook(codebook_xlsx: Path, sheet: str) -> pd.DataFrame:
    cb_raw = pd.read_excel(codebook_xlsx, sheet_name=sheet)
    lemma_col, pos_col, cat_col = find_codebook_cols(cb_raw)

    cb = cb_raw[[lemma_col, pos_col, cat_col]].copy()
    cb.columns = ["lemma", "pos_group", "category"]

    cb["lemma"] = cb["lemma"].astype(str).str.strip().str.lower()
    cb["pos_group"] = cb["pos_group"].apply(normalize_pos_group)
    cb["category"] = cb["category"].astype(str).str.strip().str.upper()

    cb = cb[cb["category"].notna() & (cb["category"] != "")]
    cb = cb.drop_duplicates(subset=["lemma"])  # codebook is single truth per lemma

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

    # Ensure numeric head/word_id
    tok["word_id"] = pd.to_numeric(tok["word_id"], errors="coerce").fillna(-1).astype(int)
    tok["head"] = pd.to_numeric(tok["head"], errors="coerce").fillna(-1).astype(int)

    return tok


def restrict_to_codebook(tok: pd.DataFrame, cb: pd.DataFrame, clusters: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Inner-join to restrict token universe to codebook lemmas only.
    Adds category and codebook pos_group.
    Adds cluster label.
    """
    df = tok.merge(cb, on="lemma", how="inner", suffixes=("", "_cb"))
    df = df.merge(clusters, on="transcript_id", how="left")

    cluster_col = detect_cluster_col(df)
    if cluster_col not in df.columns:
        df["group"] = "UNKNOWN"
        cluster_col = "group"

    df["token_count"] = 1
    return df, cluster_col


def category_shares(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    group_col can be 'section' or cluster label column
    Returns counts + shares within group_col.
    """
    g = df.groupby([group_col, "category"], as_index=False)["token_count"].sum()
    tot = df.groupby([group_col], as_index=False)["token_count"].sum().rename(columns={"token_count": f"{group_col}_total"})
    g = g.merge(tot, on=group_col, how="left")
    g[f"share_in_{group_col}"] = g["token_count"] / g[f"{group_col}_total"]
    return g


def pos_by_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["pos_group", "category"], as_index=False)["token_count"].sum()
    tot = df.groupby(["pos_group"], as_index=False)["token_count"].sum().rename(columns={"token_count": "pos_total"})
    g = g.merge(tot, on="pos_group", how="left")
    g["share_within_pos"] = g["token_count"] / g["pos_total"]
    return g


def distinctive_lemmas_option_c(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """
    Option C: distinctiveness of lemma within its category vs background (all codebook tokens).
    Score = log P(lemma|cat) - log P(lemma|all)
    with add-alpha smoothing.
    """
    # counts per lemma and category
    lc = df.groupby(["category", "pos_group", "lemma"], as_index=False)["token_count"].sum()
    # totals
    cat_tot = df.groupby(["category"], as_index=False)["token_count"].sum().rename(columns={"token_count": "cat_total"})
    all_total = df["token_count"].sum()
    vocab_size = df["lemma"].nunique()

    lc = lc.merge(cat_tot, on="category", how="left")
    lc["all_total"] = all_total
    lc["V"] = vocab_size

    # P(lemma|cat)
    lc["p_in_cat"] = (lc["token_count"] + alpha) / (lc["cat_total"] + alpha * vocab_size)

    # Need P(lemma|all): counts of lemma across all categories
    l_all = df.groupby(["lemma"], as_index=False)["token_count"].sum().rename(columns={"token_count": "lemma_total"})
    lc = lc.merge(l_all, on="lemma", how="left")
    lc["p_in_all"] = (lc["lemma_total"] + alpha) / (all_total + alpha * vocab_size)

    lc["distinctiveness"] = (lc["p_in_cat"].apply(safe_log) - lc["p_in_all"].apply(safe_log))

    # Higher = more category-specific (within codebook universe)
    lc = lc.sort_values(["category", "pos_group", "distinctiveness"], ascending=[True, True, False])
    return lc


def radar_scores(distinct_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Per category: mean distinctiveness of top N lemmas (across POS combined).
    """
    tmp = distinct_df.groupby(["category"], as_index=False).apply(
        lambda g: pd.Series({
            "radar_score_mean_topN": float(g.head(top_n)["distinctiveness"].mean()),
            "radar_score_sum_topN": float(g.head(top_n)["distinctiveness"].sum()),
        })
    ).reset_index(drop=True)
    return tmp.sort_values("category")


# -------------------------
# Dependency-based "who acts" (subjects/agents)
# -------------------------

SUBJ_DEPRELS = {"nsubj", "csubj", "nsubj:pass", "csubj:pass"}
OBJ_DEPRELS = {"obj", "iobj"}
AGENT_DEPRELS = {"obl:agent", "agent"}  # UD varies; include both

def verb_subject_object_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    df is codebook-restricted token-level with head/deprel.
    We extract edges where dependent -> head is a verb token in codebook (by lemma + upos VERB)
    Output: head_lemma, category(head), dep_lemma, dep_text, dep_deprel, role(subject/object/agent), cluster, section
    """
    # Create a per-sentence lookup: head token rows and dep token rows live in same df subset.
    # We'll join dependents to their heads within the same (transcript_id, section, sent_id).
    keys = ["transcript_id", "section", "sent_id"]
    heads = df[keys + ["word_id", "lemma", "upos", "category"]].copy()
    heads = heads.rename(columns={"word_id": "head_word_id", "lemma": "head_lemma", "upos": "head_upos", "category": "head_category"})

    deps = df[keys + ["word_id", "head", "deprel", "lemma", "text"]].copy()
    deps = deps.rename(columns={"word_id": "dep_word_id", "lemma": "dep_lemma", "text": "dep_text"})

    # match dep.head -> head.word_id
    merged = deps.merge(
        heads,
        left_on=keys + ["head"],
        right_on=keys + ["head_word_id"],
        how="inner",
    )

    # only edges pointing to verbs (head_upos == VERB)
    merged = merged[merged["head_upos"].str.upper().eq("VERB")].copy()

    # classify edge role
    def edge_role(dep_rel: str) -> str:
        r = dep_rel.strip()
        if r in SUBJ_DEPRELS:
            return "SUBJECT"
        if r in OBJ_DEPRELS:
            return "OBJECT"
        if r in AGENT_DEPRELS:
            return "AGENT"
        return "OTHER"

    merged["edge_role"] = merged["deprel"].apply(edge_role)
    return merged


def agent_shares_by_category(edges: pd.DataFrame, df: pd.DataFrame, cluster_col: str,
                            ai_set: set, human_set: set) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute subject/agent share by category (HUMAN/AI/OTHER) for verbs in that category.
    Also outputs top subject lemmas by category.
    """
    # add cluster info to edges (from df)
    # easiest: merge using transcript_id + section + sent_id + dep_word_id
    keys = ["transcript_id", "section", "sent_id"]
    dep_keyed = df[keys + ["word_id", cluster_col]].rename(columns={"word_id": "dep_word_id"})
    edges = edges.merge(dep_keyed, on=keys + ["dep_word_id"], how="left")

    # focus on SUBJECT and AGENT edges
    focus = edges[edges["edge_role"].isin(["SUBJECT", "AGENT"])].copy()
    focus["agent_class"] = focus.apply(lambda r: classify_agent(r["dep_lemma"], r["dep_text"], ai_set, human_set), axis=1)

    focus["edge_count"] = 1

    shares = focus.groupby(["head_category", "agent_class"], as_index=False)["edge_count"].sum()
    tot = focus.groupby(["head_category"], as_index=False)["edge_count"].sum().rename(columns={"edge_count": "total_edges"})
    shares = shares.merge(tot, on="head_category", how="left")
    shares["share"] = shares["edge_count"] / shares["total_edges"]
    shares = shares.rename(columns={"head_category": "category"}).sort_values(["category", "share"], ascending=[True, False])

    # top subject lemmas per category
    top_subj = (
        focus.groupby(["head_category", "agent_class", "dep_lemma"], as_index=False)["edge_count"].sum()
        .sort_values(["head_category", "agent_class", "edge_count"], ascending=[True, True, False])
        .rename(columns={"head_category": "category", "dep_lemma": "subject_lemma"})
    )

    return shares, top_subj


# -------------------------
# Plotting
# -------------------------

def plot_stacked_share(df: pd.DataFrame, index_col: str, cat_col: str, share_col: str,
                       title: str, outpath: Path) -> None:
    piv = df.pivot_table(index=index_col, columns=cat_col, values=share_col, aggfunc="sum", fill_value=0)
    plt.figure(figsize=(12, 6))
    piv.plot(kind="bar", stacked=True)
    plt.title(title)
    plt.ylabel("Share")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.show()

def plot_bar(df: pd.DataFrame, index_col: str, value_col: str, title: str, outpath: Path) -> None:
    s = df.set_index(index_col)[value_col]
    plt.figure(figsize=(12, 5))
    s.plot(kind="bar")
    plt.title(title)
    plt.ylabel(value_col)
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.show()

def plot_radar(categories: List[str], scores: List[float], title: str, outpath: Path) -> None:
    """
    Radar/spider plot with matplotlib polar axes.
    """
    if len(categories) == 0:
        return

    # close the loop
    angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]
    values = scores + scores[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.1)

    plt.title(title, y=1.08)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.show()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_parquet", default="stanza_out/token_level.parquet")
    ap.add_argument("--codebook_xlsx", default="analysis/targets_with_categories.xlsx")
    ap.add_argument("--codebook_sheet", default="codebook")
    ap.add_argument("--clusters_tsv", default="interviews_clustered.tsv")
    ap.add_argument("--outdir", default="analysis/categorized")
    ap.add_argument("--role", default="user", help="user | all")
    ap.add_argument("--alpha", type=float, default=0.5, help="Smoothing for distinctiveness")
    ap.add_argument("--topN", type=int, default=12, help="Top-N distinctive lemmas to list per category")
    ap.add_argument("--radar_topN", type=int, default=10, help="Top-N used to compute radar scores")
    ap.add_argument("--ai_lex", default="", help="Optional path to txt file: one AI lemma per line")
    ap.add_argument("--human_lex", default="", help="Optional path to txt file: one HUMAN lemma per line")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    ensure_dir(outdir)
    ensure_dir(figdir)

    # Load resources
    cb = load_codebook(Path(args.codebook_xlsx), args.codebook_sheet)
    cl = load_clusters(Path(args.clusters_tsv))
    tok = load_tokens(Path(args.token_parquet), role=args.role)

    # Restrict to codebook + attach clusters
    df, cluster_col = restrict_to_codebook(tok, cb, cl)

    # -------------------------
    # Core descriptive tables
    # -------------------------
    by_section = category_shares(df, "section")
    by_section.to_csv(outdir / "category_by_section.csv", index=False)

    by_cluster = category_shares(df, cluster_col)
    by_cluster.to_csv(outdir / "category_by_cluster.csv", index=False)

    pos_cat = pos_by_category(df)
    pos_cat.to_csv(outdir / "pos_by_category.csv", index=False)

    # Lemmas in each category (inventory + counts)
    lemma_cat_counts = (
        df.groupby(["category", "pos_group", "lemma"], as_index=False)["token_count"].sum()
        .sort_values(["category", "pos_group", "token_count"], ascending=[True, True, False])
    )
    lemma_cat_counts.to_csv(outdir / "lemma_by_category_counts.csv", index=False)

    # -------------------------
    # (1) Distinctive lexis per category (Option C) + Radar
    # -------------------------
    distinct = distinctive_lemmas_option_c(df, alpha=args.alpha)
    distinct.to_csv(outdir / "distinctiveness_optionC_full.csv", index=False)

    top_distinct = distinct.groupby(["category", "pos_group"], as_index=False).head(args.topN)
    top_distinct.to_csv(outdir / "top_distinctive_lemmas_by_category.csv", index=False)

    radar = radar_scores(distinct, top_n=args.radar_topN)
    radar.to_csv(outdir / "radar_scores_by_category.csv", index=False)

    # radar plot (use mean_topN)
    cats = radar["category"].tolist()
    scores = radar["radar_score_mean_topN"].tolist()
    plot_radar(cats, scores, f"Category distinctiveness (mean top {args.radar_topN})", figdir / "radar_category_distinctiveness.png")

    # -------------------------
    # (2) Who controls/authors/ideates: subject/agent attribution by category
    # -------------------------
    ai_set = set(DEFAULT_AI_LEMMAS)
    human_set = set(DEFAULT_HUMAN_LEMMAS)

    if args.ai_lex:
        ai_set |= {norm_lower(x) for x in Path(args.ai_lex).read_text(encoding="utf-8").splitlines() if x.strip()}
    if args.human_lex:
        human_set |= {norm_lower(x) for x in Path(args.human_lex).read_text(encoding="utf-8").splitlines() if x.strip()}

    edges = verb_subject_object_edges(df)
    edges.to_csv(outdir / "verb_dependency_edges_full.csv", index=False)

    agent_shares, top_subjects = agent_shares_by_category(edges, df, cluster_col, ai_set, human_set)
    agent_shares.to_csv(outdir / "agent_shares_by_category.csv", index=False)
    top_subjects.to_csv(outdir / "top_subjects_by_category_full.csv", index=False)

    # Plot agent shares (stacked by category)
    # We want: x=category, stacked shares for HUMAN/AI/OTHER
    agent_piv = agent_shares.pivot_table(index="category", columns="agent_class", values="share", aggfunc="sum", fill_value=0)
    plt.figure(figsize=(10, 5))
    agent_piv.plot(kind="bar", stacked=True)
    plt.title("Who is the SUBJECT/AGENT of category verbs?")
    plt.ylabel("Share")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "agent_shares_by_category.png", dpi=220)
    plt.show()

    # Optional: top subjects per category (top 15 per agent_class)
    top_subj15 = top_subjects.sort_values(["category", "agent_class", "edge_count"], ascending=[True, True, False]) \
                             .groupby(["category", "agent_class"]).head(15)
    top_subj15.to_csv(outdir / "top_subjects_by_category_top15.csv", index=False)

    # -------------------------
    # (3) Distributions by cluster + distinctive lexis by cluster within category
    # -------------------------
    plot_stacked_share(by_cluster, cluster_col, "category", f"share_in_{cluster_col}",
                       "Category share by profession cluster (codebook-only)", figdir / "category_share_by_cluster.png")

    plot_stacked_share(by_section, "section", "category", "share_in_section",
                       "Category share by section (codebook-only)", figdir / "category_share_by_section.png")

    # Distinctive lemmas by (cluster, category) within codebook universe:
    # score = log P(lemma|cluster,cat) - log P(lemma|cluster)
    # (still Option C style, but conditioned on cluster)
    df_cc = df.copy()
    # counts
    cc = df_cc.groupby([cluster_col, "category", "pos_group", "lemma"], as_index=False)["token_count"].sum()
    # totals
    cc_tot = df_cc.groupby([cluster_col, "category"], as_index=False)["token_count"].sum().rename(columns={"token_count": "cc_total"})
    c_tot = df_cc.groupby([cluster_col], as_index=False)["token_count"].sum().rename(columns={"token_count": "c_total"})
    l_cluster = df_cc.groupby([cluster_col, "lemma"], as_index=False)["token_count"].sum().rename(columns={"token_count": "lemma_in_cluster"})
    V = df_cc["lemma"].nunique()
    alpha = args.alpha

    cc = cc.merge(cc_tot, on=[cluster_col, "category"], how="left")
    cc = cc.merge(c_tot, on=[cluster_col], how="left")
    cc = cc.merge(l_cluster, on=[cluster_col, "lemma"], how="left")

    cc["p_lemma_given_cluster_cat"] = (cc["token_count"] + alpha) / (cc["cc_total"] + alpha * V)
    cc["p_lemma_given_cluster"] = (cc["lemma_in_cluster"] + alpha) / (cc["c_total"] + alpha * V)
    cc["distinctiveness_cluster"] = cc["p_lemma_given_cluster_cat"].apply(safe_log) - cc["p_lemma_given_cluster"].apply(safe_log)

    cc = cc.sort_values([cluster_col, "category", "pos_group", "distinctiveness_cluster"], ascending=[True, True, True, False])
    cc.to_csv(outdir / "distinctiveness_by_cluster_by_category_full.csv", index=False)

    cc.groupby([cluster_col, "category", "pos_group"]).head(min(12, args.topN)).to_csv(
        outdir / "top_distinctive_lemmas_by_cluster_by_category.csv", index=False
    )

    # -------------------------
    # Tiny report (markdown-ish tables as CSVs)
    # -------------------------
    # A concise “what nouns/verbs are in which category” inventory
    inv = cb.sort_values(["category", "pos_group", "lemma"])
    inv.to_csv(outdir / "codebook_inventory_by_category.csv", index=False)

    print("\nWrote outputs to:", outdir)
    print("Figures in:", figdir)
    print("Key files:")
    print(" - category_by_section.csv / category_by_cluster.csv / pos_by_category.csv")
    print(" - top_distinctive_lemmas_by_category.csv + radar_category_distinctiveness.png")
    print(" - agent_shares_by_category.csv + agent_shares_by_category.png")
    print(" - top_distinctive_lemmas_by_cluster_by_category.csv")


if __name__ == "__main__":
    main()
