from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import csv

pca = PCA(n_components=100)

"""  usage:
python scripts/ratio_calculation.py --triplets stanza_out/kg2/edges_ext_all.csv --manual_trips triplet_codebook.tsv --out results_new/similarity_by_section_manual_triplets.tsv --model all-MiniLM-L6-v2 --save_sentences --group_by transcript_id,section --jobs interviews_by_job.tsv

# incl. PCA-reduction:
# python scripts/ratio_calculation.py --triplets stanza_out/kg2/edges_ext_all.csv --manual_trips triplet_codebook.tsv --out results_new/similarities_by_section_manual_triplets_pca.tsv --model all-MiniLM-L6-v2 --save_sentences --group_by transcript_id,section --jobs interviews_by_job.tsv --pca

"""

COLS = [
    "transcript_id", "role", "section", "subsection",
    "sent_id", "token_id",
    "text", "lemma", "upos", "xpos", "feats",
    "head", "deprel", "misc"
]
N_COLS = len(COLS)
relevant_sections = ['basic_job_description','walkthrough','project_example','dynamic']

def read_token_csv_loose(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
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


def load_jobs(filepath: str) -> pd.DataFrame:
    """
    Load transcript_id -> creative_type mapping from a TSV file.
    Expected columns: transcript_id, creative_type
    """
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    df['transcript_id'] = df['transcript_id'].str.strip()
    df['creative_type'] = df['creative_type'].str.strip()
    return df[['transcript_id', 'creative_type']]


def reconstruct_sentences(df: pd.DataFrame) -> pd.DataFrame:
    sentences = []
    groupby_cols = ["transcript_id", "section", "sent_id"]

    for group_keys, sent_df in df.groupby(groupby_cols, dropna=False):
        sent_df = sent_df.sort_values("word_id")

        tokens = []
        for _, row in sent_df.iterrows():
            token_text = str(row["text"])

            if not token_text or token_text == "nan":
                continue

            if token_text.startswith("'") and tokens:
                tokens[-1] = tokens[-1] + token_text
            else:
                tokens.append(token_text)

        sentence_text = " ".join(tokens).strip()

        if sentence_text:
            sentences.append({
                "transcript_id": group_keys[0],
                "role": group_keys[1],
                "section": group_keys[2],
                "subsection": group_keys[3],
                "sent_id": group_keys[4],
                "text": sentence_text
            })

    return pd.DataFrame(sentences)


def compute_sentence_embeddings(sentences_df: pd.DataFrame,
                                model_name: str = " ", do_pca=False) -> pd.DataFrame:
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    texts = sentences_df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    sentences_df['embedding'] = [emb for emb in embeddings]
    if do_pca:
        embeddings_pca = pca.fit_transform(embeddings)
        sentences_df['embedding_pca'] = [emb for emb in embeddings_pca]
    return sentences_df

def compute_triplet_embeddings(triplet_df: pd.DataFrame,
                                model_name: str = " ",
                               text_column: str = "text",do_pca=False) -> pd.DataFrame:
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    texts = triplet_df[text_column].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if do_pca:
        embeddings_pca = pca.fit_transform(embeddings)
        triplet_df['embedding_pca'] = [emb for emb in embeddings_pca]
    triplet_df['embedding'] = [emb for emb in embeddings]
    print(f"Computed embeddings for {len(triplet_df)} sentences / triplets.")
    return triplet_df

def average_hum_ai_vectors(vectors_df: pd.DataFrame,pca_reduced=False):
    vectors_df.drop_duplicates(subset=['raw_text'])
    human_embs = np.vstack(vectors_df[vectors_df['agent_binary'] == 'HUMAN']['embedding'].values)
    ai_embs = np.vstack(vectors_df[vectors_df['agent_binary'] == 'AI']['embedding'].values)

    human_vector = np.mean(human_embs, axis=0)
    ai_vector = np.mean(ai_embs, axis=0)
    if pca_reduced:
        human_vector = pca.transform([human_vector])[0]
        ai_vector = pca.transform([ai_vector])[0]
    return human_vector, ai_vector


def compute_similarities_by_section(
    df: pd.DataFrame,
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    group_by: str = 'section',
    embedding_col: str = 'embedding',
    text_col: str = 'text'
) -> Dict:
    results = {}
    group_cols = [col.strip() for col in group_by.split(',')]

    for group_keys, section_df in df.groupby(group_cols):
        if len(group_cols) == 1:
            group_name = str(group_keys)
        else:
            group_name = '_'.join(str(k) for k in group_keys)

        embeddings = np.vstack(section_df[embedding_col].values)

        sim_to_a = cosine_similarity(embeddings, vector_a.reshape(1, -1)).flatten()
        sim_to_b = cosine_similarity(embeddings, vector_b.reshape(1, -1)).flatten()

        idx_closest_a = np.argmax(sim_to_a)
        idx_closest_b = np.argmax(sim_to_b)

        section_list = section_df.reset_index(drop=True)
        closest_to_a = section_list.iloc[idx_closest_a]
        closest_to_b = section_list.iloc[idx_closest_b]

        results[group_name] = {
            'n_sentences': len(section_df),

            'closest_to_a': {
                'text': closest_to_a[text_col],
                'similarity': float(sim_to_a[idx_closest_a]),
                'sent_id': int(closest_to_a.get('sent_id', idx_closest_a)) if pd.notna(closest_to_a.get('sent_id')) else None,
                'transcript_id': str(closest_to_a.get('transcript_id', '')),
            },
            'avg_similarity_to_a': float(np.mean(sim_to_a)),
            'std_similarity_to_a': float(np.std(sim_to_a)),
            'min_similarity_to_a': float(np.min(sim_to_a)),
            'max_similarity_to_a': float(np.max(sim_to_a)),

            'closest_to_b': {
                'text': closest_to_b[text_col],
                'similarity': float(sim_to_b[idx_closest_b]),
                'sent_id': int(closest_to_b.get('sent_id', idx_closest_b)) if pd.notna(closest_to_b.get('sent_id')) else None,
                'transcript_id': str(closest_to_b.get('transcript_id', '')),
            },
            'avg_similarity_to_b': float(np.mean(sim_to_b)),
            'std_similarity_to_b': float(np.std(sim_to_b)),
            'min_similarity_to_b': float(np.min(sim_to_b)),
            'max_similarity_to_b': float(np.max(sim_to_b)),

            'all_similarities_to_a': sim_to_a.tolist(),
            'all_similarities_to_b': sim_to_b.tolist(),
        }

    return results


def create_summary_dataframe(results: Dict) -> pd.DataFrame:
    summary_rows = []

    for section, stats in results.items():
        summary_rows.append({
            'section': section,
            'n_sentences': stats['n_sentences'],

            'avg_sim_to_a': stats['avg_similarity_to_a'],
            'std_sim_to_a': stats['std_similarity_to_a'],
            'min_sim_to_a': stats['min_similarity_to_a'],
            'max_sim_to_a': stats['max_similarity_to_a'],
            'closest_to_a_text': stats['closest_to_a']['text'],
            'closest_to_a_sim': stats['closest_to_a']['similarity'],
            'closest_to_a_sent_id': stats['closest_to_a']['sent_id'],

            'avg_sim_to_b': stats['avg_similarity_to_b'],
            'std_sim_to_b': stats['std_similarity_to_b'],
            'min_sim_to_b': stats['min_similarity_to_b'],
            'max_sim_to_b': stats['max_similarity_to_b'],
            'closest_to_b_text': stats['closest_to_b']['text'],
            'closest_to_b_sim': stats['closest_to_b']['similarity'],
            'closest_to_b_sent_id': stats['closest_to_b']['sent_id'],
        })

    return pd.DataFrame(summary_rows)


def load_triplets_as_sentences(filepath: str) -> pd.DataFrame:
    if filepath.endswith('.tsv'):
        df = pd.read_csv(filepath,sep='\t')
    else:
        df = pd.read_csv(filepath)
    df.fillna("", inplace=True)
    return df

def load_manual_triplets_as_sentences(filepath: str) -> pd.DataFrame:
    ## COLS :
    # agent_binary	subj_lemma	pred_lemma	obj_lemma	action_lemma
    # pattern	voice	ai_in_obl	ai_obl_hits	freq
    # n_transcripts	n_sections	n_clusters	triplet_sig	examples	examples_ctx	ai_in_ctx
    df_manual_triplets = pd.read_csv(filepath, sep='\t')
    df_manual_triplets = df_manual_triplets[df_manual_triplets['agent_binary'].notna()]
    df_manual_triplets.fillna("", inplace=True)
    return df_manual_triplets




def print_results(results: Dict, label: str = ""):
    header = f" [{label}]" if label else ""
    for section, stats in results.items():
        print(f"\n{'=' * 80}")
        print(f"SECTION: {section}{header}")
        print(f"{'=' * 80}")
        print(f"Number of sentences: {stats['n_sentences']}")

        print(f"\n--- Vector A ---")
        print(f"Average similarity: {stats['avg_similarity_to_a']:.4f} (±{stats['std_similarity_to_a']:.4f})")
        print(f"Range: [{stats['min_similarity_to_a']:.4f}, {stats['max_similarity_to_a']:.4f}]")
        print(f"Closest sentence (sim={stats['closest_to_a']['similarity']:.4f}):")
        print(f"  {stats['closest_to_a']['text']}")

        print(f"\n--- Vector B ---")
        print(f"Average similarity: {stats['avg_similarity_to_b']:.4f} (±{stats['std_similarity_to_b']:.4f})")
        print(f"Range: [{stats['min_similarity_to_b']:.4f}, {stats['max_similarity_to_b']:.4f}]")
        print(f"Closest sentence (sim={stats['closest_to_b']['similarity']:.4f}):")
        print(f"  {stats['closest_to_b']['text']}")


def save_results(results: Dict, out_path: Path, fmt: str):
    """Save a results dict to CSV, parquet, or JSON."""
    summary_df = create_summary_dataframe(results)

    if fmt == "csv":
        summary_df.to_csv(out_path, index=False)
    elif fmt == "parquet":
        summary_df.to_parquet(out_path, index=False)
    else:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Always also save detailed JSON alongside
    detailed_path = out_path.parent / (out_path.stem + "_detailed_results_similarities.json")
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Summary  -> {out_path}")
    print(f"  Detailed -> {detailed_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute sentence similarities to reference vectors by section")
    ap.add_argument("--tokens", required=False, help="Path to token-level CSV file")
    ap.add_argument("--manual_trips", required=True,
                    help="Path to manually labelled triplets (.tsv)")
    ap.add_argument("--out", required=True, help="Output path for results")
    ap.add_argument("--model", default="all-MiniLM-L6-v2",
                    help="Sentence transformer model name (default: all-MiniLM-L6-v2)")
    ap.add_argument("--section_col", default="section", help="Column name for sections")
    ap.add_argument("--format", default="csv", choices=["csv", "json", "parquet"],
                    help="Output format")
    ap.add_argument("--pca", default=False,
                    help="Apply dimensionality reduction?",action="store_true")
    ap.add_argument("--save_sentences", action="store_true",
                    help="Save reconstructed sentences to a separate file")
    ap.add_argument("--sentences_file", type=str,
                    help="Path to reconstructed sentences.")
    ap.add_argument("--group_by", default="section",
                    help="Grouping column(s): 'section', 'transcript_id', or 'transcript_id,section'")
    ap.add_argument("--triplets", required=False, help="Path to SVO triplets TSV file")
    ap.add_argument("--jobs", required=False,
                    help="Path to interviews_by_job.tsv with columns transcript_id, creative_type. "
                         "When provided, separate result files are written per creative_type.")
    args = ap.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    # (data to be embedded / clustered / )
    if args.triplets:
        print(f"Loading triplets from {args.triplets}...")
        df_sentences = load_triplets_as_sentences(args.triplets)
        print(f"  Loaded {len(df_sentences)} triplets in columns {df_sentences.keys()}")
        print(f"  Sections: {df_sentences[args.section_col].unique().tolist()}")
        # ── Embed (once, for all data) ─────────────────────────────────────────────
        df_sentences['raw_text'] = df_sentences['raw_text'].apply(lambda x: x.replace('_',''))
        df_sentences = df_sentences[df_sentences['section'].isin(relevant_sections)].reset_index(drop=True)
        print(f"  Sections: {df_sentences[args.section_col].unique().tolist()}")
        df_sentences = compute_triplet_embeddings(df_sentences, model_name=args.model,text_column='raw_text',do_pca=args.pca)
    else:
        print(f"Loading token-level data from {args.tokens}...")
        df_tokens = read_token_csv_loose(args.tokens)
        print(f"  Loaded {len(df_tokens)} tokens")
        df_sentences = reconstruct_sentences(df_tokens)
        print(f"  Reconstructed {len(df_sentences)} sentences")
        df_sentences = df_sentences[df_sentences['section'].isin(relevant_sections)].reset_index(drop=True)
        print(f"  Sections: {df_sentences[args.section_col].unique().tolist()}")
        df_sentences = compute_triplet_embeddings(df_sentences, model_name=args.model, text_column='text',do_pca=args.pca)

    ### manually in console:
    # df_sentences = load_triplets_as_sentences("stanza_out/kg2/edges_ext_all.csv")
    # print(f"  Loaded {len(df_sentences)} triplets in columns {df_sentences.keys()}")
    # print(f"  Sections: {df_sentences['section'].unique().tolist()}")
    # df_sentences['raw_text'] = df_sentences['raw_text'].apply(lambda x: x.replace('_',''))
    # # ── Embed (once, for all data) ─────────────────────────────────────────────
    # df_sentences = compute_triplet_embeddings(df_sentences, model_name='all-MiniLM-L6-v2', text_column='raw_text',do_pca=True)
    # manual_df  = load_manual_triplets_as_sentences('triplet_codebook.tsv')

    if args.save_sentences:
        sent_path = Path(args.out).parent / "reconstructed_sentences.csv"
        df_sentences.to_csv(sent_path, index=False)
        print(f"  Saved reconstructed sentences to: {sent_path}")




    # ── Load manually labelled triplets ──────────────────────────────────────────
    print(f"\nLoading reference triplets from {args.manual_trips}...")

    manual_df  = load_manual_triplets_as_sentences(args.manual_trips)

    manual_df['_key'] = range(len(manual_df))

    merged1 = pd.merge(manual_df, df_sentences,
                       left_on=['subj_lemma', 'pred_lemma', 'obj_lemma'],
                       right_on=['subj_lemma', 'predicate_lemma', 'obj_lemma'],
                       how="inner")

    merged2 = pd.merge(manual_df, df_sentences,
                       left_on=['subj_lemma', 'pred_lemma', 'action_lemma'],
                       right_on=['subj_lemma', 'predicate_lemma', 'obj_lemma'],
                       how="inner")
    # Keys that matched in either merge
    matched_keys = set(merged1['_key']).union(set(merged2['_key']))
    # Unmatched rows
    unmatched_df = manual_df[~manual_df['_key'].isin(matched_keys)].drop(columns='_key')
    # Cleanup
    manual_df.drop(columns='_key', inplace=True)
    #partial_match = pd.merge(unmatched_df,df_sentences,left_on=['subj_lemma','pred_lemma'],right_on=['subj_lemma','predicate_lemma'],how="inner")
    manual_triplet_embeddings = pd.concat([merged1,merged2])

    ## different depending on triplet-based or sentence-based similarity calcution?

    vector_a_full, vector_b_full = average_hum_ai_vectors(manual_triplet_embeddings, pca_reduced=False)
    print(f"  Vector A shape: {vector_a_full.shape}")
    print(f"  Vector B shape: {vector_b_full.shape}")


    if args.pca:
        vector_a, vector_b = average_hum_ai_vectors(manual_triplet_embeddings, pca_reduced=args.pca)
        print(f"  Vector A shape: {vector_a.shape}")
        print(f"  Vector B shape: {vector_b.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)


    type_out_merged_ = out_path.parent / "joint_manual_all_triplets.tsv"

    manual_triplet_embeddings.to_csv(type_out_merged_, sep='\t')


    # ── Run & save ────────────────────────────────────────────────────────────
    if args.jobs:
        # Join creative_type at the summary stage: split df_sentences by type,
        # run the full similarity pipeline per group, write separate files.
        print(f"\nLoading job/creative_type mapping from {args.jobs}...")
        jobs_df = load_jobs(args.jobs)
        print(f"  Found {jobs_df['creative_type'].nunique()} creative types: "
              f"{sorted(jobs_df['creative_type'].unique().tolist())}")

        # Left-join so unmatched transcripts are flagged, not silently dropped
        df_with_type = df_sentences.merge(
            jobs_df, on='transcript_id', how='left'
        )

        unmatched = df_with_type['creative_type'].isna().sum()
        if unmatched:
            print(f"  WARNING: {unmatched} rows have no matching creative_type "
                  f"and will be written to a separate '_unmatched' file.")

        groups = df_with_type.groupby('creative_type', dropna=False)

        for creative_type, group_df in groups:
            type_label = str(creative_type) if pd.notna(creative_type) else "unmatched"
            print(f"\n{'─' * 60}")
            print(f"Processing creative_type: {type_label}  ({len(group_df)} rows)")

            results = compute_similarities_by_section(
                group_df,
                vector_a_full,
                vector_b_full,
                group_by=args.group_by,
                embedding_col='embedding',
                text_col='raw_text'
            )

            print_results(results, label=type_label)

            # Build per-type output path, e.g. results/similarity_writer.csv
            type_out = out_path.parent / f"{out_path.stem}_{type_label}{out_path.suffix}"
            print(f"\nSaving results for '{type_label}':")
            save_results(results, type_out, args.format)
        if args.pca:
            for creative_type, group_df in groups:
                type_label = str(creative_type) if pd.notna(creative_type) else "unmatched"
                print(f"\n{'─' * 60}")
                print(f"Processing creative_type: {type_label}  ({len(group_df)} rows)")

                results = compute_similarities_by_section(
                    group_df,
                    vector_a,
                    vector_b,
                    group_by=args.group_by,
                    embedding_col='embedding_pca',
                    text_col='raw_text'
                )
                print_results(results, label=type_label)
                # Build per-type output path, e.g. results/similarity_writer.csv
                type_out = out_path.parent / f"{out_path.stem}_{type_label}_PCAd{out_path.suffix}"
                print(f"\nSaving results for '{type_label}_PCAd':")
                save_results(results, type_out, args.format)

    # Original behaviour: single output for all data
    print(f"\nComputing similarities by section for all jobs...")
    results = compute_similarities_by_section(
        df_sentences,
        vector_a_full,
        vector_b_full,
        group_by=args.group_by,
        embedding_col='embedding',
        text_col='raw_text'
    )

    print_results(results)
    print(f"\n{'=' * 80}")
    print("Saving results:")
    out_all = out_path.parent / f"{out_path.stem}_all_{out_path.suffix}"

    save_results(results, out_all, args.format)

    print(f"\n{'=' * 80}")

    if args.pca:
        # Original behaviour: single output for all data
        print(f"\nComputing similarities by section for all jobs, PCA-reduced...")
        results = compute_similarities_by_section(
            df_sentences,
            vector_a,
            vector_b,
            group_by=args.group_by,
            embedding_col='embedding_pca',
            text_col='raw_text'
        )

        print_results(results)
        print(f"\n{'=' * 80}")
        print("Saving results:")
        out_all = out_path.parent / f"{out_path.stem}_all_PCAd{out_path.suffix}"

        save_results(results, out_all, args.format)

        print(f"\n{'=' * 80}")

    ######### now with the bertopic script
    from bertopic_bytype import *
    from bertopic import BERTopic
    import pandas as pd
    import numpy as np

    import seaborn as sns
    from pathlib import Path
    from sklearn.metrics.pairwise import cosine_similarity

    #df = pd.read_csv('interview_split.tsv', sep='\t', index_col=0)
    #df['all_text'] = df[user_cols].fillna('').astype(str).agg(''.join, axis=1)
    # old bertopic model
    topic_model = BERTopic.load("./BERTriplet_model")

    all_triplets = df_sentences
    topics_pred, probs_pred = topic_model.transform(all_triplets['raw_text'])

    all_triplets['topic_BERTopic'] = topics_pred
    all_triplets['topic_BERTopic_probs'] = probs_pred
    # on newly formatted triplets
    # topic_model_new = BERTopic()
    # trips = all_triplets['raw_text']
    # topics_new, probs_new = topic_model_new.fit_transform(trips)
    # topic_model_new.save("BERTriplet_model_NEW")

    # ── 3. Compute cosine similarities to topic centres (once) ───────────────────

    topic_centers = topic_model.topic_embeddings_[1:]  # skip outlier topic -1
    embeddings = topic_model.embedding_model.embedding_model.encode(
        all_triplets['raw_text'].tolist()
    )
    pca2 = PCA(n_components=100)

    for opt in ['bertopic','manual_vectors']:
        if opt == "bertopic":
            if args.pca:
                print('PCA fitting on BERTopic')
                pca_embeddings = pca.fit_transform(embeddings)
                similarities = cosine_similarity(pca_embeddings, pca.transform(topic_centers))
            else:
                similarities = cosine_similarity(embeddings,topic_centers)
        else:
            if args.pca:
                embeddings_matrix = np.vstack(all_triplets['embedding_pca'].values)
                similarities = cosine_similarity(embeddings_matrix, [vector_a,vector_b])
            else:
                embeddings_matrix = np.vstack(all_triplets['embedding'].values)
                similarities = cosine_similarity(embeddings_matrix, [vector_a_full,vector_b_full])

        all_triplets['sim_to_HUMAN'] = similarities[:, 0]
        all_triplets['sim_to_AI'] = similarities[:, 1]
        all_triplets['sim_difference_individual'] = (
                all_triplets['sim_to_HUMAN'] - all_triplets['sim_to_AI']
        )

        # Normalize once across the full dataset
        min_d = all_triplets['sim_difference_individual'].min()
        max_d = all_triplets['sim_difference_individual'].max()
        all_triplets['normalized_difference'] = (
                (all_triplets['sim_difference_individual'] - min_d) / (max_d - min_d)
        )
        all_triplets['balanced_ratio'] = (
                (all_triplets['sim_to_HUMAN'] - all_triplets['sim_to_AI']) /
                (all_triplets['sim_to_HUMAN'] + all_triplets['sim_to_AI'])
        )

        # ── 4. BERTopic ratio_df (once on full data) ──────────────────────────────────
        ratio_df_full = all_triplets.groupby(
            ['transcript_id', 'section']
        ).apply(calculate_ratios).reset_index()

        ratio_df_full['section'] = pd.Categorical(
            ratio_df_full['section'], categories=section_order_all, ordered=True
        )

        # ── 5. Load creative_type mapping ─────────────────────────────────────────────

        JOBS_FILE = args.jobs  # <-- adjust path if needed

        if args.pca:
            OUTPUT_ROOT = Path(f"results_new/{opt}_PCA")
        else:
            OUTPUT_ROOT = Path(f"results_new/{opt}")
        jobs_df = load_jobs(JOBS_FILE)
        creative_types = sorted(jobs_df['creative_type'].unique().tolist())
        print(f"\nFound creative types: {creative_types}")

        # ── 6. Run analysis: ALL data first ───────────────────────────────────────────

        all_out = make_output_dir(OUTPUT_ROOT, f"all_{opt}")

        run_analysis_for_group(all_triplets, all_out, label="", ratio_df=ratio_df_full)

        # ── 7. Run analysis: per creative_type ────────────────────────────────────────

        # Join creative_type onto all_triplets
        all_triplets_typed = all_triplets.merge(jobs_df, on='transcript_id', how='left')

        unmatched = all_triplets_typed['creative_type'].isna().sum()
        if unmatched:
            print(f"\nWARNING: {unmatched} rows have no matching creative_type → written to 'unmatched' folder.")

        for ct, group_df in all_triplets_typed.groupby('creative_type', dropna=False):
            ct_label = str(ct) if pd.notna(ct) else "unmatched"
            print(f"\n{'─' * 60}")
            print(f"Processing creative_type: {ct_label}  ({len(group_df)} rows)")

            ct_out = make_output_dir(OUTPUT_ROOT, ct_label)

            # Subset ratio_df to transcripts in this group
            ct_transcript_ids = set(group_df['transcript_id'].unique())
            ratio_df_ct = ratio_df_full[
                ratio_df_full['transcript_id'].isin(ct_transcript_ids)
            ].copy()

            run_analysis_for_group(group_df, ct_out, label=ct_label, ratio_df=ratio_df_ct)

        print(f"\n{'═' * 60}")
        print(f'\n{opt}:  ')
        print(f"Done. Results written under: {OUTPUT_ROOT}/")
        print(f"{'═' * 60}")


        if args.pca:
            all_out_trips = Path(args.out).parent / f"all_triplet_similarities_{opt}_PCA.tsv"
        else:
            all_out_trips = Path(args.out).parent / f"all_triplet_{opt}_similarities.tsv"


        all_triplets_copy = all_triplets.drop(columns=['embedding'])
        if args.pca:
            all_triplets_copy.drop(columns=['embedding_pca'], inplace=True)
        all_triplets_copy.to_csv(all_out_trips, sep='\t')

