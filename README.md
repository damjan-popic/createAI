# createAI — Stanza annotation pipeline (split interviews)

## What this does
- Takes a TSV where each interview is split into section columns (e.g. `u_*`, `a_*`)
- Runs **Stanza full English stack**: tokenize + MWT + POS + lemma + **dependency parse (deprel/head)**
- Writes **4 outputs** for analysis/sharing:
  - `token_level` (one row per word; full annotations incl. `head`, `deprel`)
  - `section_summary` (counts per transcript × role × section; incl. verb/noun counts)
  - `lemma_summary` (lemma frequencies per transcript × role × section × POS)
  - `transcript_summary` (section counts summed per transcript; role=`user`/`assistant`/`all`)
- Optional: exports **CoNLL-U** files

## Input format
- TSV with a `transcript_id` column
- Section columns:
  - user sections start with `u_` (e.g. `u_driver`)
  - assistant sections start with `a_` (e.g. `a_driver`)
- Optional accidental index column `Unnamed: 0` is ignored if present

## Dependencies
  - `stanza`
  - `pandas`
  - `tqdm`
  - `pyarrow` (for Parquet)
  - `openpyxl` (only for `--format xlsx` or `--extra xlsx`)

## CLI

python -m pip install -U stanza pandas tqdm pyarrow openpyxl

python stanza_tag_sections.py \
  --input interview_split.tsv \
  --outdir stanza_out \
  --which user \
  --extra csv