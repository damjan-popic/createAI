#!/usr/bin/env python3
"""
Stanza tagger for split interview sections.

Outputs (always produced in primary format):
  1) token_level            one row per Stanza word (has head + deprel)
  2) section_summary        one row per transcript_id × role × section
  3) lemma_summary          one row per transcript_id × role × section × lemma × upos
  4) transcript_summary     sums across sections per transcript_id, with role=user/assistant/all

Primary output format defaults to Parquet.
You can additionally request extra shareable formats (csv/tsv/jsonl/xlsx).

Optional: export CoNLL-U files with --conllu.

python stanza_tag_sections.py \
  --input interview_split.tsv \
  --outdir stanza_out \
  --which both \
  --extra csv

  dependencies: 

"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import stanza


ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
}

# Column groups in your TSV (you can extend if needed)
USER_PREFIX = "u_"
ASSIST_PREFIX = "a_"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "section"


def detect_sections(df: pd.DataFrame, which: str) -> Tuple[List[str], List[str]]:
    """Return (user_cols, assistant_cols) based on prefixes."""
    cols = list(df.columns)

    user_cols = [c for c in cols if c.startswith(USER_PREFIX)]
    asst_cols = [c for c in cols if c.startswith(ASSIST_PREFIX)]

    if which == "user":
        return user_cols, []
    if which == "assistant":
        return [], asst_cols
    if which == "both":
        return user_cols, asst_cols
    raise ValueError(f"Invalid --which: {which}")


def ensure_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def write_df(df: pd.DataFrame, path_base: Path, fmt: str) -> Path:
    """Write a dataframe to path_base with extension based on fmt. Returns the written path."""
    fmt = fmt.lower()
    if fmt == "parquet":
        outpath = path_base.with_suffix(".parquet")
        df.to_parquet(outpath, index=False)
        return outpath
    if fmt == "csv":
        outpath = path_base.with_suffix(".csv")
        df.to_csv(outpath, index=False)
        return outpath
    if fmt == "tsv":
        outpath = path_base.with_suffix(".tsv")
        df.to_csv(outpath, sep="\t", index=False)
        return outpath
    if fmt == "jsonl":
        outpath = path_base.with_suffix(".jsonl")
        df.to_json(outpath, orient="records", lines=True, force_ascii=False)
        return outpath
    if fmt == "xlsx":
        # For xlsx we expect to be writing a workbook elsewhere; this function isn't used.
        raise ValueError("xlsx writing is handled separately.")
    raise ValueError(f"Unsupported format: {fmt}")


def write_workbook_xlsx(
    outpath: Path,
    token_level: pd.DataFrame,
    section_summary: pd.DataFrame,
    lemma_summary: pd.DataFrame,
    transcript_summary: pd.DataFrame,
) -> None:
    """Write all outputs into a single Excel workbook with 4 sheets."""
    # NOTE: token_level can be huge; Excel has row limits (1,048,576).
    # If you exceed it, this will fail. That's why parquet/csv is primary.
    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        token_level.to_excel(writer, sheet_name="token_level", index=False)
        section_summary.to_excel(writer, sheet_name="section_summary", index=False)
        lemma_summary.to_excel(writer, sheet_name="lemma_summary", index=False)
        transcript_summary.to_excel(writer, sheet_name="transcript_summary", index=False)


def stanza_pipeline(lang: str = "en") -> stanza.Pipeline:
    # Full stack incl dependencies:
    # tokenize, mwt, pos, lemma, depparse
    return stanza.Pipeline(
        lang=lang,
        processors="tokenize,mwt,pos,lemma,depparse",
        tokenize_no_ssplit=False,
        use_gpu=False,  # flip if you have GPU stanza configured; safe default
        verbose=False,
    )


def iter_sections(
    row: pd.Series,
    transcript_id: str,
    user_cols: List[str],
    asst_cols: List[str],
) -> Iterable[Tuple[str, str, str]]:
    """
    Yield (role, section_name, text).
    role in {"user","assistant"}
    section_name is derived from column name.
    """
    for c in user_cols:
        text = row.get(c, "")
        yield "user", c[len(USER_PREFIX):], "" if pd.isna(text) else str(text)
    for c in asst_cols:
        text = row.get(c, "")
        yield "assistant", c[len(ASSIST_PREFIX):], "" if pd.isna(text) else str(text)


def conllu_for_doc(doc: stanza.Document) -> str:
    # Stanza supports "{:c}".format(doc) which yields CoNLL-U.
    return "{:c}".format(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input TSV file (e.g., interview_split.tsv)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--which", default="both", choices=["user", "assistant", "both"],
                    help="Process user sections, assistant sections, or both")
    ap.add_argument("--format", default="parquet",
                    choices=["parquet", "csv", "tsv", "jsonl", "xlsx"],
                    help="Primary output format (default: parquet)")
    ap.add_argument("--extra", nargs="*", default=[],
                    choices=["parquet", "csv", "tsv", "jsonl", "xlsx"],
                    help="Also write additional formats (e.g., --extra csv tsv). "
                         "Parquet remains canonical if chosen as primary.")
    ap.add_argument("--include_text", action="store_true",
                    help="Include raw text in section_summary (can increase size)")
    ap.add_argument("--conllu", action="store_true",
                    help="Additionally export CoNLL-U files per transcript_id/role/section")
    ap.add_argument("--conllu_dir", default="conllu",
                    help="Subdirectory under outdir for CoNLL-U files (default: conllu)")
    ap.add_argument("--lang", default="en", help="Language for stanza (default: en)")
    args = ap.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    ensure_output_dir(outdir)

    # Load TSV (your file is TSV)
    df = pd.read_csv(inpath, sep="\t")
    # Drop accidental index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "transcript_id" not in df.columns:
        raise ValueError("Expected a 'transcript_id' column in the input TSV.")

    user_cols, asst_cols = detect_sections(df, args.which)

    nlp = stanza_pipeline(args.lang)

    token_rows: List[Dict] = []
    section_rows: List[Dict] = []

    conllu_outdir = outdir / args.conllu_dir
    if args.conllu:
        ensure_output_dir(conllu_outdir)

    # Process
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcripts"):
        tid = str(row["transcript_id"])

        for role, section_name, text in iter_sections(row, tid, user_cols, asst_cols):
            section_key = slugify(section_name)
            text_clean = (text or "").strip()

            if not text_clean:
                # still output empty summaries so you keep shape consistent
                section_rec = {
                    "transcript_id": tid,
                    "role": role,
                    "section": section_key,
                    "section_raw": section_name,
                    "n_sents": 0,
                    "n_words": 0,
                    "n_verbs": 0,
                    "n_aux": 0,
                    "n_verbs_total": 0,
                    "n_nouns": 0,
                    "n_propn": 0,
                    "n_nouns_total": 0,
                }
                if args.include_text:
                    section_rec["text"] = text
                section_rows.append(section_rec)
                continue

            doc = nlp(text_clean)

            # Optional CoNLL-U export
            if args.conllu:
                conllu_txt = conllu_for_doc(doc)
                outname = f"{tid}__{role}__{section_key}.conllu"
                (conllu_outdir / outname).write_text(conllu_txt, encoding="utf-8")

            # Gather token-level rows + POS counts
            n_sents = len(doc.sentences)
            n_words = 0
            n_verbs = 0
            n_aux = 0
            n_nouns = 0
            n_propn = 0

            for si, sent in enumerate(doc.sentences, start=1):
                for wi, w in enumerate(sent.words, start=1):
                    n_words += 1
                    upos = w.upos or ""
                    if upos == "VERB":
                        n_verbs += 1
                    elif upos == "AUX":
                        n_aux += 1
                    elif upos == "NOUN":
                        n_nouns += 1
                    elif upos == "PROPN":
                        n_propn += 1

                    token_rows.append({
                        "transcript_id": tid,
                        "role": role,
                        "section": section_key,
                        "section_raw": section_name,
                        "sent_id": si,
                        "word_id": wi,
                        "text": w.text,
                        "lemma": w.lemma,
                        "upos": w.upos,
                        "xpos": w.xpos,
                        "feats": w.feats,
                        "head": w.head,      # int, 0=root
                        "deprel": w.deprel,  # dependency relation
                    })

            section_rec = {
                "transcript_id": tid,
                "role": role,
                "section": section_key,
                "section_raw": section_name,
                "n_sents": n_sents,
                "n_words": n_words,
                "n_verbs": n_verbs,
                "n_aux": n_aux,
                "n_verbs_total": n_verbs + n_aux,
                "n_nouns": n_nouns,
                "n_propn": n_propn,
                "n_nouns_total": n_nouns + n_propn,
            }
            if args.include_text:
                section_rec["text"] = text_clean
            section_rows.append(section_rec)

    token_df = pd.DataFrame(token_rows)
    section_df = pd.DataFrame(section_rows)

    # Lemma summary: transcript × role × section × lemma × upos
    if len(token_df) > 0:
        lemma_df = (
            token_df
            .dropna(subset=["lemma", "upos"])
            .groupby(["transcript_id", "role", "section", "lemma", "upos"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
    else:
        lemma_df = pd.DataFrame(columns=["transcript_id", "role", "section", "lemma", "upos", "count"])

    # Transcript summary: sums across sections per transcript_id, role + all
    if len(section_df) > 0:
        numeric_cols = [
            "n_sents", "n_words",
            "n_verbs", "n_aux", "n_verbs_total",
            "n_nouns", "n_propn", "n_nouns_total",
        ]
        transcript_by_role = (
            section_df
            .groupby(["transcript_id", "role"], as_index=False)[numeric_cols]
            .sum()
        )
        transcript_all = (
            section_df
            .groupby(["transcript_id"], as_index=False)[numeric_cols]
            .sum()
        )
        transcript_all.insert(1, "role", "all")
        transcript_df = pd.concat([transcript_by_role, transcript_all], ignore_index=True)
    else:
        transcript_df = pd.DataFrame(columns=["transcript_id", "role"])

    # Write outputs in primary format
    primary = args.format.lower()
    extras = [e.lower() for e in args.extra if e.lower() != primary]

    # Special-case XLSX: single workbook; but still write primary dataframes if parquet/csv etc
    if primary == "xlsx":
        xlsx_path = outdir / "stanza_outputs.xlsx"
        write_workbook_xlsx(xlsx_path, token_df, section_df, lemma_df, transcript_df)
    else:
        write_df(token_df, outdir / "token_level", primary)
        write_df(section_df, outdir / "section_summary", primary)
        write_df(lemma_df, outdir / "lemma_summary", primary)
        write_df(transcript_df, outdir / "transcript_summary", primary)

    # Write extras
    for fmt in extras:
        if fmt == "xlsx":
            xlsx_path = outdir / "stanza_outputs.xlsx"
            write_workbook_xlsx(xlsx_path, token_df, section_df, lemma_df, transcript_df)
        else:
            write_df(token_df, outdir / f"token_level__{fmt}", fmt)
            write_df(section_df, outdir / f"section_summary__{fmt}", fmt)
            write_df(lemma_df, outdir / f"lemma_summary__{fmt}", fmt)
            write_df(transcript_df, outdir / f"transcript_summary__{fmt}", fmt)


if __name__ == "__main__":
    main()
