# createAI

## Output
  - `token_level` – ena vrstica na besedo (vključno s `head`, `deprel`)
  - `section_summary` – statistike po intervjujih × vloga × sekcija (glagoli/samostalniki)
  - `lemma_summary` – frekvence lem po sekcijah
  - `transcript_summary` – seštevki po intervjujih
- 
Po želji izvozi **CoNLL-U**

## Vhod
- TSV z obveznim stolpcem `transcript_id`
- Sekcije:
  - uporabnik: `u_*`
  - AI: `a_*`


## Dependencies 
python -m pip install -U stanza pandas tqdm pyarrow openpyxl

## How to run

python stanza_tag_sections.py \
  --input interview_split.tsv \
  --outdir stanza_out \
  --which both \
  --extra csv
## Flags
--which : user | assistant | both
--format : primarni izhod (privzeto parquet)
--extra : dodatni formati (csv, tsv, …)
--include_text : vključi surovo besedilo
--conllu : izvozi CoNLL-U
