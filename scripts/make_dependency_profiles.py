#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

INPATH = Path("stanza_out/token_level.parquet")
OUTDIR = Path("analysis")
OUTDIR.mkdir(exist_ok=True)

# How many targets?
TOP_N = 200
TOP_K_ITEMS = 12      # how many items per list field
N_EXAMPLES = 4        # examples per lemma

# Which relations to summarize for verbs ... hmmmm
VERB_DEPRELS = ["nsubj", "nsubj:pass", "obj", "iobj", "ccomp", "xcomp", "obl"]

# Which relations to summarize for nouns (noun as dependent)
NOUN_DEPRELS = ["obj", "iobj", "nsubj", "nsubj:pass", "obl", "nmod", "amod", "compound"]

def topk_str(counter: Counter, k: int) -> str:
    return " | ".join([f"{w}:{c}" for w, c in counter.most_common(k)])

def load_token():
    df = pd.read_parquet(INPATH)
    # drop missing
    df = df.dropna(subset=["lemma", "upos", "deprel", "head"])
    # normalize types
    df["head"] = df["head"].astype(int)
    df["word_id"] = df["word_id"].astype(int)
    return df

tok = load_token()

# --- choose top 200 verbs and nouns by overall frequency (within token table)

freq = tok.groupby(["lemma", "upos"], as_index=False).size().rename(columns={"size":"freq"})
top_verbs = (freq[freq["upos"]=="VERB"].sort_values("freq", ascending=False).head(TOP_N))["lemma"].tolist()
top_nouns = (freq[freq["upos"].isin(["NOUN","PROPN"])].sort_values("freq", ascending=False).head(TOP_N))["lemma"].tolist()


# sentence key
sent_key = ["transcript_id","role","section","sent_id"]
sent_text = (
    tok.sort_values(sent_key+["word_id"])
       .groupby(sent_key)["text"]
       .apply(lambda xs: " ".join(xs.astype(str)))
       .reset_index()
       .rename(columns={"text":"sentence"})
)


heads = tok[sent_key+["word_id","lemma","upos"]].rename(
    columns={"word_id":"head_word_id","lemma":"head_lemma","upos":"head_upos"}
)

deps = tok.copy().rename(columns={"lemma":"dep_lemma","upos":"dep_upos","word_id":"dep_word_id"})

# match dependent.head -> head.word_id within same sentence
dep_with_head = deps.merge(
    heads,
    left_on=sent_key+["head"],
    right_on=sent_key+["head_word_id"],
    how="left"
).drop(columns=["head_word_id"])

# attach sentence strings
dep_with_head = dep_with_head.merge(sent_text, on=sent_key, how="left")

# ------------------------
# VERB PROFILES (verb is head)
# ------------------------
verb_rows = []
for v in top_verbs:
    sub = dep_with_head[(dep_with_head["head_lemma"] == v) & (dep_with_head["deprel"].isin(VERB_DEPRELS))]

    # counters by deprel
    ctr = {r: Counter(sub[sub["deprel"]==r]["dep_lemma"].tolist()) for r in VERB_DEPRELS}

    # pick examples where the verb occurs in the sentence
    ex_df = dep_with_head[(dep_with_head["dep_lemma"] == v) & (dep_with_head["dep_upos"]=="VERB")]
    ex = ex_df["sentence"].dropna().drop_duplicates().head(N_EXAMPLES).tolist()

    verb_rows.append({
        "lemma": v,
        "subjects_top": topk_str(ctr["nsubj"] + ctr["nsubj:pass"], TOP_K_ITEMS),
        "objects_top": topk_str(ctr["obj"] + ctr["iobj"], TOP_K_ITEMS),
        "xcomp_top": topk_str(ctr["xcomp"], TOP_K_ITEMS),
        "ccomp_top": topk_str(ctr["ccomp"], TOP_K_ITEMS),
        "obl_top": topk_str(ctr["obl"], TOP_K_ITEMS),
        "examples": " || ".join(ex),
    })

verb_profiles = pd.DataFrame(verb_rows)
verb_profiles.to_csv(OUTDIR / "dep_profiles_verbs.csv", index=False)

# ------------------------
# NOUN PROFILES (noun is dependent)
# ------------------------
noun_rows = []
for n in top_nouns:
    sub = dep_with_head[(dep_with_head["dep_lemma"] == n) & (dep_with_head["dep_upos"].isin(["NOUN","PROPN"]))]

    # role of noun in dependency tree (as dependent)
    role_ctr = Counter(sub["deprel"].tolist())

    # head lemma where noun is obj/nsubj/obl etc.
    gov = sub[sub["head_upos"]=="VERB"]
    gov_ctr = Counter(gov["head_lemma"].dropna().tolist())

    # adjectives modifying noun: amod where head is noun lemma
    amod = dep_with_head[(dep_with_head["head_lemma"] == n) & (dep_with_head["deprel"]=="amod")]
    amod_ctr = Counter(amod["dep_lemma"].tolist())

    # compounds modifying noun
    comp = dep_with_head[(dep_with_head["head_lemma"] == n) & (dep_with_head["deprel"]=="compound")]
    comp_ctr = Counter(comp["dep_lemma"].tolist())

    # examples where noun occurs
    ex_df = sub
    ex = ex_df["sentence"].dropna().drop_duplicates().head(N_EXAMPLES).tolist()

    noun_rows.append({
        "lemma": n,
        "noun_roles_top": topk_str(role_ctr, TOP_K_ITEMS),
        "head_verbs_top": topk_str(gov_ctr, TOP_K_ITEMS),
        "amod_top": topk_str(amod_ctr, TOP_K_ITEMS),
        "compound_top": topk_str(comp_ctr, TOP_K_ITEMS),
        "examples": " || ".join(ex),
    })

noun_profiles = pd.DataFrame(noun_rows)
noun_profiles.to_csv(OUTDIR / "dep_profiles_nouns.csv", index=False)

print("Wrote:")
print(OUTDIR / "dep_profiles_verbs.csv")
print(OUTDIR / "dep_profiles_nouns.csv")
