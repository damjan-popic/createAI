#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INPATH = Path("stanza_out/lemma_summary.parquet")
OUTDIR = Path("analysis")
OUTDIR.mkdir(exist_ok=True)

# Load lemma-level summary
lem = pd.read_parquet(INPATH)

# Top 100 NOUN+PROPN per interview (transcript_id)
nouns = (
    lem[lem["upos"].isin(["NOUN", "PROPN"])]
    .groupby(["transcript_id", "lemma"], as_index=False)["count"]
    .sum()
)

nouns["rank"] = nouns.groupby("transcript_id")["count"] \
                     .rank(method="dense", ascending=False)

top100_nouns = (
    nouns[nouns["rank"] <= 100]
    .sort_values(["transcript_id", "rank", "lemma"])
)

# Optional: annotation column
top100_nouns["category"] = ""

# Save
top100_nouns.to_parquet(OUTDIR / "top100_nouns_per_interview.parquet", index=False)
top100_nouns.to_csv(OUTDIR / "top100_nouns_per_interview.csv", index=False)

print("Wrote:")
print(OUTDIR / "top100_nouns_per_interview.parquet")
print(OUTDIR / "top100_nouns_per_interview.csv")