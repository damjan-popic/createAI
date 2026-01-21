#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

INPATH = Path("stanza_out/lemma_summary.parquet")
OUTDIR = Path("analysis")
OUTDIR.mkdir(exist_ok=True)

lem = pd.read_parquet(INPATH)

verbs = (
    lem[lem["upos"] == "VERB"]
    .groupby(["transcript_id", "section", "lemma"], as_index=False)["count"]
    .sum()
)

verbs["rank"] = (
    verbs
    .groupby(["transcript_id", "section"])["count"]
    .rank(method="dense", ascending=False)
)

top100_verbs = (
    verbs[verbs["rank"] <= 100]
    .sort_values(["transcript_id", "section", "rank", "lemma"])
)

# Column for later manual annotation, perhaps
top100_verbs["category"] = ""

top100_verbs.to_parquet(OUTDIR / "top100_verbs_per_section.parquet", index=False)
top100_verbs.to_csv(OUTDIR / "top100_verbs_per_section.csv", index=False)

print("Wrote:")
print(OUTDIR / "top100_verbs_per_section.parquet")
print(OUTDIR / "top100_verbs_per_section.csv")
