#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INPATH = Path("stanza_out/lemma_summary.parquet")
OUTDIR = Path("analysis")
OUTDIR.mkdir(exist_ok=True)

lem = pd.read_parquet(INPATH)

nouns = (
    lem[lem["upos"].isin(["NOUN", "PROPN"])]
    .groupby(["transcript_id", "section", "lemma"], as_index=False)["count"]
    .sum()
)

nouns["rank"] = (
    nouns
    .groupby(["transcript_id", "section"])["count"]
    .rank(method="dense", ascending=False)
)

top100_nouns = (
    nouns[nouns["rank"] <= 100]
    .sort_values(["transcript_id", "section", "rank", "lemma"])
)

top100_nouns["category"] = ""

top100_nouns.to_parquet(OUTDIR / "top100_nouns_per_section.parquet", index=False)
top100_nouns.to_csv(OUTDIR / "top100_nouns_per_section.csv", index=False)

print("Wrote:")
print(OUTDIR / "top100_nouns_per_section.parquet")
print(OUTDIR / "top100_nouns_per_section.csv")
