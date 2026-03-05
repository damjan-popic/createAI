from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

"""
usage:

python scripts/extract_triplets_kg.py  --in stanza_out/token_level__csv.csv --exclude_sections intro --out_dir stanza_out/kg2 --format csv

"""

# ----- schema (adjust if needed) -----
COLS = [
    "transcript_id", "role", "section", "subsection",
    "sent_id", "token_id",
    "text", "lemma", "upos", "xpos", "feats",
    "head", "deprel", "misc"
]

N_COLS = len(COLS)

STOP_SECTIONS_DEFAULT = {"intro"}  # exclude these


def slugify_pred(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s or "rel"


def stable_id(prefix: str, *parts: str) -> str:
    h = hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


def read_token_csv_loose(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"', escapechar="\\")
        for raw in reader:
            if not raw:
                continue
            trimmed = (raw[:N_COLS] + [""] * N_COLS)[:N_COLS]
            rows.append(trimmed)

    df = pd.DataFrame(rows, columns=COLS)

    for c in ["sent_id", "token_id", "head"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def build_sentence_index(sent_df: pd.DataFrame):
    tok = {int(r["token_id"]): r for _, r in sent_df.iterrows() if pd.notna(r["token_id"])}
    children: Dict[int, List[int]] = {}
    for tid, r in tok.items():
        hid = int(r["head"]) if pd.notna(r["head"]) else 0
        children.setdefault(hid, []).append(tid)
    return tok, children


def subtree_tokens(root_id: int, children: Dict[int, List[int]]) -> List[int]:
    out = []
    stack = [root_id]
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        for ch in children.get(x, []):
            stack.append(ch)
    return sorted(out)

SAFE_MODS = {
    "det", "amod", "compound", "nummod", "appos", "flat", "name",
    "nmod:poss", "case"  # 'case' we will usually EXCLUDE from surface span, but it's here for reference
}

# Relations to NEVER expand into for "short objects"
BLOCK_EXPAND_PREFIX = {
    "acl", "advcl", "ccomp", "conj", "parataxis", "discourse", "punct", "cc"
}

def rel_blocked(rel: str) -> bool:
    if rel in BLOCK_EXPAND_PREFIX:
        return True
    for p in BLOCK_EXPAND_PREFIX:
        if rel.startswith(p + ":"):
            return True
    return False

def children_by_rel(head_id, tok, children) -> Dict[str, list[int]]:
    out = {}
    for ch in children.get(head_id, []):
        r = str(tok[ch]["deprel"])
        out.setdefault(r, []).append(ch)
    return out

def compact_nominal_span(root_id: int, tok, children,
                         max_pp_depth: int = 0,
                         exclude_case_tokens: bool = True) -> list[int]:
    """
    Build a short NP span:
    - include root
    - include det/amod/compound/nummod/name/flat/appos/nmod:poss
    - optionally include ONE PP layer (obl/nmod) with depth control
    - never include relative clauses / ccomp / xcomp / conj etc.
    -- now include xcomp!!
    - optionally exclude the preposition token (case) so object isn't "for AI"
    """
    keep = {root_id}
    stack = [root_id]
    pp_depth = {root_id: 0}

    while stack:
        hid = stack.pop()
        for ch in children.get(hid, []):
            rel = str(tok[ch]["deprel"])
            if rel_blocked(rel):
                continue

            # don't include case tokens in object surface span
            if exclude_case_tokens and (rel == "case" or rel.startswith("case:")):
                continue

            # safe local modifiers
            if rel in {"det","amod","compound","nummod","appos","flat","name","nmod:poss"}:
                keep.add(ch)
                stack.append(ch)
                pp_depth[ch] = pp_depth.get(hid, 0)
                continue

            # optional limited PP expansion (keep it SHORT)
            if rel in {"nmod", "obl"} or rel.startswith("nmod:") or rel.startswith("obl:"):
                d = pp_depth.get(hid, 0) + 1
                if d <= max_pp_depth:
                    keep.add(ch)
                    stack.append(ch)
                    pp_depth[ch] = d
                continue

            # other relations: ignore by default

    return sorted(keep)

def compact_clause_span(clause_root_id: int, tok, children,
                        exclude_case_tokens: bool = True,
                        include_mark: bool = True) -> list[int]:
    """
    Build a short clause: (optional subj) + mark ("to") + aux + main verb + compact obj
    """
    keep = {clause_root_id}

    # Include infinitival "to" marker
    if include_mark:
        for ch in children.get(clause_root_id, []):
            rel = str(tok[ch]["deprel"])
            if rel == "mark" or rel.startswith("mark:"):
                keep.add(ch)

    # auxiliaries / negation / particles that make the predicate readable
    for ch in children.get(clause_root_id, []):
        rel = str(tok[ch]["deprel"])
        if rel in {"aux", "aux:pass", "neg", "compound:prt"}:
            keep.add(ch)

    # subject of the embedded clause (if any)
    for ch in children.get(clause_root_id, []):
        rel = str(tok[ch]["deprel"])
        if rel == "nsubj" or rel.startswith("nsubj:"):
            keep.update(compact_nominal_span(ch, tok, children, max_pp_depth=0, exclude_case_tokens=exclude_case_tokens))

    # object (prefer obj then obl)
    obj = None
    for relname in ("obj", "iobj"):
        for ch in children.get(clause_root_id, []):
            if str(tok[ch]["deprel"]) == relname:
                obj = ch
                break
        if obj is not None:
            break

    action_tok_id = None
    for relname in ("xcomp"):
        for ch in children.get(clause_root_id, []):
            if str(tok[ch]["deprel"]) == relname:
                action_tok_id = ch
                break


    if obj is not None:
        keep.update(compact_nominal_span(obj, tok, children, max_pp_depth=0, exclude_case_tokens=exclude_case_tokens))
    else:
        # allow a short obl if there is no obj
        for ch in children.get(clause_root_id, []):
            if str(tok[ch]["deprel"]).startswith("obl"):
                keep.update(compact_nominal_span(ch, tok, children, max_pp_depth=0, exclude_case_tokens=exclude_case_tokens))
                break
    if action_tok_id is not None:
        keep.update(compact_nominal_span(action_tok_id, tok, children, max_pp_depth=0, exclude_case_tokens=exclude_case_tokens))

    return sorted(keep)


def span_text(tok_ids: List[int], tok: Dict[int, pd.Series]) -> str:
    parts = []
    prev_no_space = False

    for tid in tok_ids:
        upos = str(tok[tid]["upos"])
        if upos == "PUNCT":
            continue

        text = str(tok[tid]["text"])
        misc = str(tok[tid].get("misc", ""))

        # Attach without space if previous token said so, or if starts with apostrophe
        if (prev_no_space or text.startswith("'")) and parts:
            parts[-1] = parts[-1] + text
        else:
            parts.append(text)

        # Check if next token should attach
        prev_no_space = "SpaceAfter=No" in misc

    return " ".join(parts).strip()




EXCLUDE_IN_OBJECT = {
    "nsubj", "nsubj:pass",
    "cop", "aux", "aux:pass",
    "punct", "discourse",
    "parataxis", "cc", 'case'
}
def subtree_tokens_filtered(root_id: int, tok, children, exclude_rels: set[str]) -> list[int]:
    out = []
    stack = [root_id]
    seen = set()
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        for ch in children.get(x, []):
            rel = str(tok[ch]["deprel"])
            # exclude both exact and prefix forms (e.g. nsubj:pass)
            if rel in exclude_rels or any(rel.startswith(r + ":") for r in exclude_rels):
                continue
            stack.append(ch)
    return sorted(out)


def find_children(head_id: int, tok: Dict[int, pd.Series], children: Dict[int, List[int]], deprel_prefix: str) -> List[int]:
    out = []
    for ch in children.get(head_id, []):
        rel = str(tok[ch]["deprel"])
        if rel == deprel_prefix or rel.startswith(deprel_prefix + ":"):
            out.append(ch)
    return out


def get_case_marker(obl_id: int, tok: Dict[int, pd.Series], children: Dict[int, List[int]]) -> Optional[str]:
    # In UD, case marker is usually a child of the nominal with deprel="case"
    case_children = find_children(obl_id, tok, children, "case")
    if not case_children:
        return None
    # take first case marker token text (e.g., "for", "with", "in")
    return str(tok[case_children[0]]["lemma"] or tok[case_children[0]]["text"]).lower()


@dataclass
class Edge:
    edge_id: str
    subj_id: str
    obj_id: str

    # KG-friendly predicate (lemma-based, normalized)
    predicate_lemma: str
    subj_lemma: str
    obj_lemma: str
   #Surface predicate (original token text, incl. case marker if any)
    subj_text: str
    predicate_text: str
    obj_text: str

    transcript_id: str
    role: str
    section: str
    subsection: str
    sent_id: int

    voice: str
    pattern: str
    raw_text : str
    action_text: str = ""
    action_lemma: str = ""

KEEP_PREMOD_RELS = {"det", "amod", "compound", "nummod", "flat", "name", "nmod:poss"}
DROP_RELS_PREFIX = {"acl", "advcl", "ccomp", "conj", "parataxis", "discourse", "cc", "punct"}
DROP_RELS_EXACT = {"nmod", "case", "mark"}  # <- kills "for X", "to me", "assuming that ..."

def rel_blocked(rel: str) -> bool:
    if rel in DROP_RELS_EXACT:
        return True
    if rel in DROP_RELS_PREFIX:
        return True
    return any(rel.startswith(p + ":") for p in DROP_RELS_PREFIX)

def short_np_span(root_id: int, tok, children) -> list[int]:
    """
    Head + premodifiers only. No PPs, no clausal/post modifiers.
    Also excludes prepositions (case) so you don't get 'for ghost writing'.
    """
    keep = {root_id}
    stack = [root_id]

    while stack:
        hid = stack.pop()
        for ch in children.get(hid, []):
            rel = str(tok[ch]["deprel"])
            if rel_blocked(rel):
                continue
            if rel in KEEP_PREMOD_RELS:
                keep.add(ch)
                stack.append(ch)

    return sorted(keep)
PRONOUN_MAP = {
    "i": "USER",
    "me": "USER",
    "my": "USER",
    "mine": "USER",
    "we": "USER_GROUP",
    "us": "USER_GROUP",
    "our": "USER_GROUP",
    "ours": "USER_GROUP",
}

def simplify_subject(sid: int, tok, children,
                     mode: str = "compact",
                     map_pronouns: bool = True) -> tuple[str, str]:
    """
    Returns (subj_text, subj_id_string_basis)
    subj_id_string_basis should be the canonical string you want to hash into an ID.
    """

    # Mode C: canonical pronouns
    if map_pronouns and str(tok[sid]["upos"]) == "PRON":
        key = str(tok[sid]["lemma"] or tok[sid]["text"]).lower()
        if key in PRONOUN_MAP:
            canon = PRONOUN_MAP[key]
            return canon, canon.lower()

    if mode == "head":
        t = str(tok[sid]["text"]).strip()
        return t, t.lower()

    # mode == "compact"
    ids = compact_nominal_span(
        sid, tok, children,
        max_pp_depth=0,             # subjects: keep it tight
        exclude_case_tokens=True
    )
    text = span_text(ids, tok)

    # if compact span ends up empty (rare), fall back
    if not text:
        text = str(tok[sid]["text"]).strip()

    return text, text.lower()

def extract_edges_from_sentence(sent_df: pd.DataFrame, topverbs: Optional[list] = None) -> List[Edge]:
    tok, children = build_sentence_index(sent_df)
    if not tok:
        return []

    meta = sent_df.iloc[0]
    transcript_id = str(meta["transcript_id"])
    speaker = str(meta["role"])
    section = str(meta["section"])
    subsection = str(meta["subsection"])
    sent_id = int(meta["sent_id"]) if pd.notna(meta["sent_id"]) else -1

    # Predicates = VERB + copular heads (that have 'cop' child)
    predicate_ids = []
    for tid, r in tok.items():
        if str(r["upos"]) == "VERB":
            predicate_ids.append(tid)
        else:
            if find_children(tid, tok, children, "cop"):
                predicate_ids.append(tid)

    edges: List[Edge] = []
    edge_ids = list()

    for pid in predicate_ids:
        pr = tok[pid]
        pid_lemma = str(pr["lemma"]).lower()
        if topverbs:
            if topverbs and pid_lemma not in topverbs:
                continue

        # subjects
        subjs = find_children(pid, tok, children, "nsubj")
        pass_subjs = find_children(pid, tok, children, "nsubj:pass")

        voice = "active"
        use_subjs = subjs
        if pass_subjs:
            voice = "passive"
            use_subjs = pass_subjs

        if not use_subjs:
            continue

        # relation base: copular uses cop lemma, else predicate lemma
        cop_children = find_children(pid, tok, children, "cop")
        if cop_children:
            cop_id = cop_children[0]
            rel_base_lemma = (str(tok[cop_id]["lemma"]).lower() or "be")
            rel_base_text = (str(tok[cop_id]["text"]).lower() or rel_base_lemma)
            pattern_base = "cop"
            head_complement_id = pid
        else:
            rel_base_lemma = (str(pr["lemma"]).lower() or "")

            # For text version: include aux tokens (e.g., "have used")
            aux_children = find_children(pid, tok, children, "aux")
            aux_pass_children = find_children(pid, tok, children, "aux:pass")
            all_aux = aux_children # + aux_pass_children

            if all_aux:
                # Sort by token position to get correct order
                all_aux_sorted = sorted(all_aux, key=lambda x: x)
                aux_parts = [str(tok[aid]["text"]).lower() for aid in all_aux_sorted]
                pred_text_part = str(pr["text"]).lower()
                rel_base_text = " ".join(aux_parts + [pred_text_part])
            else:
                rel_base_text = (str(pr["text"]).lower() or rel_base_lemma)

            pattern_base = "verb"
            head_complement_id = None

        # targets
        objs = find_children(pid, tok, children, "obj")
        iobjs = find_children(pid, tok, children, "iobj")
        obls = find_children(pid, tok, children, "obl")
        ccomps = find_children(pid, tok, children, "ccomp")
        xcomps = find_children(pid, tok, children, "xcomp")



        targets: List[Tuple[str, int]] = []

        for oid in objs:
            targets.append(("obj", oid))

        #if not targets:
        for oid in iobjs:
            targets.append(("iobj", oid))

        # if no core objects, allow:
        #if not targets:
            # copular: use the head itself as complement (e.g., I am good)
        if head_complement_id is not None:
            targets.append(("pred", head_complement_id))

        # then obls as secondary edges
        #if not targets:
        for oid in obls:
            targets.append(("obl", oid))

        # then clausal complements
        #if not targets:
        for oid in ccomps:
            targets.append(("ccomp", oid))
        for oid in xcomps:
            targets.append(("xcomp", oid))



        for sid in use_subjs:
            subject_lemma = str(tok[sid]["lemma"])
            subj_text, subj_basis = simplify_subject(
                sid, tok, children,
                mode="compact",
                map_pronouns=False
            )
            subj_id = stable_id("ent", subj_basis)
            subj_ids = set(compact_nominal_span(sid, tok, children, max_pp_depth=0, exclude_case_tokens=True))

            if not subj_text:
                continue

            if len(targets) < 1:
                targets.append(("no_obj", -1))

            absorbed_xcomps = set()
            for oid in objs:
                for xid in xcomps:
                    obj_text = span_text(compact_nominal_span(oid, tok, children, max_pp_depth=0), tok)
                    xcomp_ids = compact_clause_span(xid, tok, children, exclude_case_tokens=True, include_mark=True)
                    xcomp_text = span_text(xcomp_ids, tok)

                    if not obj_text or not xcomp_text:
                        continue

                    absorbed_xcomps.add(xid)


            for role, oid in targets:
                # Skip xcomp edges that are already captured in an obj_xcomp edge
                if role == "xcomp" and oid in absorbed_xcomps:
                    continue
                # Skip if object is the same as subject (common in copular sentences)
                obj_upos = ""
                if oid == sid:
                    continue

                if role == 'no_obj':
                    object_lemma =""
                else:
                    object_lemma = str(tok[oid]["lemma"])
                    obj_upos = str(tok[oid]["upos"])

                if role in {"ccomp", "xcomp"} or obj_upos in {"VERB", "AUX"}:
                    obj_ids = compact_clause_span(oid, tok, children, exclude_case_tokens=True,include_mark=True)
                elif role in ['no_obj']:
                    obj_ids = []
                else:
                    # short noun phrase; no PP expansion by default
                    obj_ids = compact_nominal_span(oid, tok, children, max_pp_depth=1, exclude_case_tokens=True)


                # Also skip if object span overlaps significantly with subject span
                obj_ids_set = set(obj_ids)
                if len(subj_ids & obj_ids_set) > 0:
                    continue

                if len(obj_ids)> 0:
                    obj_text = span_text(obj_ids, tok)

                    if not obj_text:
                        continue
                    obj_id = stable_id("ent", obj_text.lower())
                else:
                    obj_id = stable_id("ent", "")
                    obj_text =""
                # predicate normalization:
                pred_lemma = rel_base_lemma
                pred_text = rel_base_text

                if role == "obl":
                    # lemma + surface case markers (usually same, but keep both)
                    cm_lemma = get_case_marker(oid, tok, children)  # returns lemma/text-ish; good enough for EN
                    # also get surface case marker token text, if you want a true surface version:
                    case_children = find_children(oid, tok, children, "case")
                    cm_text = None
                    if case_children:
                        cm_text = str(tok[case_children[0]]["text"]).lower()

                    if cm_lemma:
                        pred_lemma = f"{pred_lemma}_{cm_lemma}"
                    if cm_text:
                        pred_text = f"{pred_text}_{cm_text}"

                pred_lemma = slugify_pred(pred_lemma)
                pred_text = slugify_pred(pred_text)

                pattern = f"{pattern_base}+{role}"


                edge_id = stable_id(
                    "edge",
                    transcript_id, role, section, subsection, str(sent_id),
                    subj_id, pred_lemma, obj_id, pattern, voice
                )

                if not pred_lemma == "bereave":
                    if edge_id not in edge_ids:
                        raw_text = " ".join([subj_text, pred_text, obj_text])
                        raw_text = raw_text.replace('_'," ")
                        raw_text = raw_text.replace(" '", "'")
                        edges.append(
                            Edge(
                                edge_id=edge_id,
                                subj_id=subj_id,
                                obj_id=obj_id,
                                predicate_lemma=pred_lemma,
                                subj_lemma = subject_lemma,
                                obj_lemma=object_lemma,
                                predicate_text=pred_text,
                                subj_text=subj_text,
                                obj_text=obj_text,
                                transcript_id=transcript_id,
                                role=speaker,
                                section=section,
                                subsection=subsection,
                                sent_id=sent_id,
                                voice=voice,
                                pattern=pattern,
                                raw_text = raw_text
                            )
                        )
                        edge_ids.append(edge_id)
            for oid in objs:
                for xid in xcomps:
                    obj_text = span_text(compact_nominal_span(oid, tok, children, max_pp_depth=0), tok)
                    xcomp_ids = compact_clause_span(xid, tok, children, exclude_case_tokens=True, include_mark=True)
                    xcomp_text = span_text(xcomp_ids, tok)

                    if not obj_text or not xcomp_text:
                        continue

                    action_lemma = str(tok[xid]["lemma"])
                    action_text = xcomp_text

                    edges.append(Edge(
                        edge_id=stable_id("edge", transcript_id, "obj_xcomp", str(sent_id), subj_id, pred_lemma, obj_text),
                        subj_id=subj_id,
                        obj_id=stable_id("ent", obj_text.lower()),
                        predicate_lemma=pred_lemma,
                        subj_lemma=subject_lemma,
                        obj_lemma=str(tok[oid]["lemma"]),  # e.g. "myself"
                        predicate_text=pred_text,
                        action_lemma=action_lemma,
                        action_text=action_text,
                        subj_text=subj_text,
                        obj_text=obj_text,
                        transcript_id=transcript_id,
                        role=speaker,
                        section=section,
                        subsection=subsection,
                        sent_id=sent_id,
                        voice=voice,
                        pattern="verb+obj_xcomp",
                        raw_text=f"{subj_text} {pred_text} {obj_text} {xcomp_text}"
                    ))


    return edges


def extract_kg(df_tokens: pd.DataFrame,
               speaker_filter: Optional[str] = "user",
               stop_sections: Optional[set[str]] = None,
               topverbs: Optional[list] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = df_tokens.copy()
    if stop_sections:
        df = df[~df["section"].isin(stop_sections)]

    if speaker_filter:
        df = df[df["role"] == speaker_filter]

    all_edges: List[Edge] = []
    for _, sent_df in df.groupby(["transcript_id", "role", "section", "subsection", "sent_id"], dropna=False):
        all_edges.extend(extract_edges_from_sentence(sent_df,topverbs))

    edges_df = pd.DataFrame([e.__dict__ for e in all_edges])

    # nodes from edges
    node_rows = {}
    for _, r in edges_df.iterrows():
        node_rows[r["subj_id"]] = {"node_id": r["subj_id"], "label": r["subj_text"], "type": "Entity"}
        node_rows[r["obj_id"]] = {"node_id": r["obj_id"], "label": r["obj_text"], "type": "Entity"}
    nodes_df = pd.DataFrame(list(node_rows.values()))

    return edges_df, nodes_df


def write_jsonl(df: pd.DataFrame, path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="token-level CSV (loose) or parquet")
    ap.add_argument("--out_dir", dest="out_dir", required=True, help="output directory")
    ap.add_argument("--speaker", dest="role", default="user")
    ap.add_argument("--exclude_sections", dest="exclude_sections", default="intro",
                    help="comma-separated sections/subsections to exclude (default: intro)")
    ap.add_argument("--format", dest="fmt", default="jsonl", choices=["jsonl", "csv", "parquet"],
                    help="output format for edges/nodes")
    ap.add_argument("--inverb", dest="topverbs", help="top 100 verbs")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = Path(args.inp)
    if inp.suffix.lower() == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = read_token_csv_loose(inp)

    if args.topverbs:
        top200 = Path(args.topverbs) ## analysis / targets_top200plus200.csv
        if top200.suffix.lower() == ".parquet":
            dfverb = pd.read_parquet(top200)
            topverbs = dfverb[dfverb['pos_group']=="VERB"]['lemma'].values.tolist()
        else:
            dfverb = pd.read_csv(top200)
            topverbs = dfverb[dfverb['pos_group'] == "VERB"]['lemma'].values.tolist()
    else:
        topverbs = None

    stop_sections = {s.strip() for s in args.exclude_sections.split(",") if s.strip()}

    edges_df, nodes_df = extract_kg(df, speaker_filter=args.role, stop_sections=stop_sections, topverbs = topverbs)

    # Separate obj and xcomp edges
    obj_edges = edges_df[edges_df['pattern'].str.contains('obj') & ~edges_df['pattern'].str.contains('xcomp')]
    xcomp_edges = edges_df[edges_df['pattern'] == 'verb+xcomp']

    # Join on transcript_id + sent_id + predicate + subject
    # so we link the obj and xcomp that share the same verb
    combined = pd.merge(
        obj_edges,
        xcomp_edges[['transcript_id', 'sent_id', 'subj_id', 'predicate_lemma', 'obj_lemma', 'obj_text']],
        on=['transcript_id', 'sent_id', 'subj_id', 'predicate_lemma'],
        how='left',
        suffixes=('', '_xcomp')
    )

    # Rename for clarity
    combined = combined.rename(columns={
        'obj_lemma_xcomp': 'action_lemma',
        'obj_text_xcomp': 'action_text'
    })


    ### SORT
    section_order = ['basic_job_description', 'walkthrough', 'project_example', 'dynamic',
                     'changed_aspects', 'concerns', 'future', 'extra_comments']

    # Create a categorical type with your custom order
    edges_df['section'] = pd.Categorical(edges_df['section'],
                                         categories=section_order,
                                         ordered=True)

    # Sort by transcript_id, then section, then sent_id
    edges_df = edges_df.sort_values(['transcript_id', 'section', 'sent_id']).reset_index(drop=True)

    if args.fmt == "jsonl":
        write_jsonl(edges_df, out_dir / "edges_ext_all.jsonl")
        write_jsonl(nodes_df, out_dir / "nodes_ext_all.jsonl")
    elif args.fmt == "csv":
        edges_df.to_csv(out_dir / "edges_ext_all.csv", index=False)
        nodes_df.to_csv(out_dir / "nodes_ext_all.csv", index=False)
    else:
        edges_df.to_parquet(out_dir / "edges_ext_all.parquet", index=False)
        nodes_df.to_parquet(out_dir / "nodes_ext_all.parquet", index=False)

    print(f"Wrote {len(edges_df)} edges and {len(nodes_df)} nodes to {out_dir}")
