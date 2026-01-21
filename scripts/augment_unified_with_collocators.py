#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import pandas as pd


VERB_REL_SUBJ = {"nsubj", "nsubj:pass"}
VERB_REL_OBJ  = {"obj", "iobj"}
VERB_REL_COMP = {"xcomp", "ccomp"}
VERB_REL_OBL  = {"obl"}

NOUN_AS_DEP_ROLES = {"obj", "iobj", "nsubj", "nsubj:pass", "obl", "nmod"}
NOUN_MODS = {"amod", "compound"}


def topk_fmt(counter: Counter, k: int) -> str:
    if not counter:
        return ""
    return " | ".join([f"{w}:{c}" for w, c in counter.most_common(k)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", required=True, help="Path to unified vocab CSV (e.g. analysis/unified_vocab_categorized.csv)")
    ap.add_argument("--token_level", required=True, help="Path to token_level.parquet (e.g. stanza_out/token_level.parquet)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--topk", type=int, default=5, help="How many collocators to keep (default: 5)")
    args = ap.parse_args()

    unified_path = Path(args.unified)
    token_path = Path(args.token_level)
    out_path = Path(args.out)

    uni = pd.read_csv(unified_path)
    tok = pd.read_parquet(token_path)

    # Normalize / safety
    tok = tok.dropna(subset=["lemma", "upos", "deprel", "head"])
    tok["head"] = tok["head"].astype(int)
    tok["word_id"] = tok["word_id"].astype(int)

    # Sentence key
    sent_key = ["transcript_id", "role", "section", "sent_id"]

    # Build head table: word_id -> head lemma/upos (within sentence)
    heads = tok[sent_key + ["word_id", "lemma", "upos"]].rename(
        columns={"word_id": "head_word_id", "lemma": "head_lemma", "upos": "head_upos"}
    )
    deps = tok.rename(columns={"lemma": "dep_lemma", "upos": "dep_upos", "word_id": "dep_word_id"})

    # Join dependent rows to their heads
    dwh = deps.merge(
        heads,
        left_on=sent_key + ["head"],
        right_on=sent_key + ["head_word_id"],
        how="left",
    ).drop(columns=["head_word_id"])

    # Limit work to lemmas present in unified list (much faster)
    target_lemmas = set(uni["lemma"].astype(str).tolist())

    # ----------------------------
    # VERB collocators (verb as HEAD)
    # ----------------------------
    verb_targets = set(uni.loc[uni["pos_group"].eq("VERB"), "lemma"].astype(str).tolist()) if "pos_group" in uni.columns \
                   else set(uni.loc[uni["upos"].astype(str).str.contains("VERB", na=False), "lemma"].astype(str).tolist())

    verb_rel = dwh[dwh["head_lemma"].isin(verb_targets)]

    # Precompute counters per verb lemma
    verb_rows = []
    for v, g in verb_rel.groupby("head_lemma"):
        subj_ctr = Counter(g.loc[g["deprel"].isin(VERB_REL_SUBJ), "dep_lemma"].dropna().astype(str))
        obj_ctr  = Counter(g.loc[g["deprel"].isin(VERB_REL_OBJ),  "dep_lemma"].dropna().astype(str))
        comp_ctr = Counter(g.loc[g["deprel"].isin(VERB_REL_COMP), "dep_lemma"].dropna().astype(str))
        obl_ctr  = Counter(g.loc[g["deprel"].isin(VERB_REL_OBL),  "dep_lemma"].dropna().astype(str))

        verb_rows.append({
            "lemma": v,
            "top_subjects": topk_fmt(subj_ctr, args.topk),
            "top_objects":  topk_fmt(obj_ctr,  args.topk),
            "top_comps":    topk_fmt(comp_ctr, args.topk),   # xcomp/ccomp dependents
            "top_obls":     topk_fmt(obl_ctr,  args.topk),
        })

    verb_prof = pd.DataFrame(verb_rows)

    # ----------------------------
    # NOUN collocators (noun as DEPENDENT + noun modifiers)
    # ----------------------------
    noun_targets = set(uni.loc[uni["pos_group"].eq("NOUN"), "lemma"].astype(str).tolist()) if "pos_group" in uni.columns \
                   else set(uni.loc[uni["upos"].astype(str).str.contains("NOUN", na=False), "lemma"].astype(str).tolist())

    noun_dep = dwh[dwh["dep_lemma"].isin(noun_targets) & dwh["dep_upos"].isin(["NOUN", "PROPN"])]
    noun_headverbs = noun_dep[noun_dep["head_upos"].eq("VERB")]

    noun_rows = []
    for n, g in noun_dep.groupby("dep_lemma"):
        # roles of the noun (as dependent)
        roles_ctr = Counter(g["deprel"].dropna().astype(str))

        # verbs governing the noun (head verbs)
        hv = noun_headverbs[noun_headverbs["dep_lemma"].eq(n)]
        headverb_ctr = Counter(hv["head_lemma"].dropna().astype(str))

        noun_rows.append({
            "lemma": n,
            "top_noun_roles": topk_fmt(roles_ctr, args.topk),
            "top_head_verbs": topk_fmt(headverb_ctr, args.topk),
        })

    noun_prof = pd.DataFrame(noun_rows)

    # modifiers of nouns: amod/compound where noun is HEAD
    noun_mod = dwh[dwh["head_lemma"].isin(noun_targets) & dwh["deprel"].isin(NOUN_MODS)]
    mod_rows = []
    for n, g in noun_mod.groupby("head_lemma"):
        amod_ctr = Counter(g.loc[g["deprel"].eq("amod"), "dep_lemma"].dropna().astype(str))
        comp_ctr = Counter(g.loc[g["deprel"].eq("compound"), "dep_lemma"].dropna().astype(str))
        mod_rows.append({
            "lemma": n,
            "top_amod": topk_fmt(amod_ctr, args.topk),
            "top_compound": topk_fmt(comp_ctr, args.topk),
        })
    noun_mod_prof = pd.DataFrame(mod_rows)

    # ----------------------------
    # Merge everything back onto unified
    # ----------------------------
    out = uni.copy()
    out = out.merge(verb_prof, on="lemma", how="left")
    out = out.merge(noun_prof, on="lemma", how="left")
    out = out.merge(noun_mod_prof, on="lemma", how="left")

    for c in ["top_subjects","top_objects","top_comps","top_obls",
              "top_noun_roles","top_head_verbs","top_amod","top_compound"]:
        if c in out.columns:
            out[c] = out[c].fillna("")

    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
