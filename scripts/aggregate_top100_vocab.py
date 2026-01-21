#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

OUTDIR = Path("analysis")
OUTDIR.mkdir(exist_ok=True)

verbs_path = OUTDIR / "top100_verbs_per_section.csv"
nouns_path = OUTDIR / "top100_nouns_per_section.csv"

verbs = pd.read_csv(verbs_path)
nouns = pd.read_csv(nouns_path)


def make_vocab(df: pd.DataFrame, upos_label: str) -> pd.DataFrame:
    # "Document" unit here = transcript_id × section
    df["doc_unit"] = df["transcript_id"].astype(str) + "||" + df["section"].astype(str)

    vocab = (
        df.groupby("lemma", as_index=False)
          .agg(
              upos=("lemma", lambda _: upos_label),
              doc_count=("doc_unit", "nunique"),
              total_count=("count", "sum"),
              mean_count=("count", "mean"),
              max_count=("count", "max"),
          )
          .sort_values(["doc_count", "total_count", "lemma"], ascending=[False, False, True])
          .reset_index(drop=True)
    )

    vocab["category"] = ""      
    vocab["notes"] = ""         # if needs be
    return vocab

verbs_vocab = make_vocab(verbs, "VERB")
nouns_vocab = make_vocab(nouns, "NOUN+PROPN")

verbs_vocab.to_csv(OUTDIR / "verbs_top100_vocab.csv", index=False)
nouns_vocab.to_csv(OUTDIR / "nouns_top100_vocab.csv", index=False)

print("Wrote:")
print(OUTDIR / "verbs_top100_vocab.csv")
print(OUTDIR / "nouns_top100_vocab.csv")
