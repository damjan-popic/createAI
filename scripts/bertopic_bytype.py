from bertopic import BERTopic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

###
# Constants

HUMAN_cols = ["u_intro", "u_basic job description",
             "u_walkthrough", "u_project example",
             "u_dynamic", "u_changed aspects",
             "u_concerns", "u_future", "u_extra comments"]

section_order_all = [
    'basic_job_description', 'walkthrough', 'project_example',
    'dynamic', 'changed_aspects', 'concerns', 'future', 'extra_comments'
]
section_order_selected = [
    'basic_job_description', 'walkthrough', 'project_example', 'dynamic'
]

green = sns.color_palette('colorblind')[2]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jobs(filepath: str) -> pd.DataFrame:
    """Load transcript_id -> creative_type mapping from a TSV file."""
    df = pd.read_csv(filepath, sep='\t', dtype=str)
    df['transcript_id'] = df['transcript_id'].str.strip()
    df['creative_type'] = df['creative_type'].str.strip()
    return df[['transcript_id', 'creative_type']]


def make_output_dir(base_dir: str, label: str) -> Path:
    """Create and return a subdirectory for a given creative_type label."""
    p = Path(base_dir) / label
    p.mkdir(parents=True, exist_ok=True)
    return p


def apply_section_order(df: pd.DataFrame, sections: list, col: str = 'section') -> pd.DataFrame:
    df = df[df[col].isin(sections)].copy()
    df[col] = pd.Categorical(df[col], categories=sections, ordered=True)
    return df


def calculate_ratios(group: pd.DataFrame) -> pd.Series:
    topics = group['topic_BERTopic'].values
    probs = group['topic_BERTopic_probs'].values

    count_0 = (topics == 0).sum() + 1
    count_1 = (topics == 1).sum() + 1

    weighted_0 = probs[topics == 0].sum() + 1
    weighted_1 = probs[topics == 1].sum() + 1

    return pd.Series({
        'simple_ratio': count_0 / count_1 / (count_0 + count_1),
        'weighted_ratio': weighted_0 / weighted_1 / (count_0 + count_1),
        'topic_0_count': count_0 - 1,
        'topic_1_count': count_1 - 1
    })


# ── Plotting functions

def plot_average_topic_counts(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        topic_0_count=('topic_0_count', 'mean'),
        topic_1_count=('topic_1_count', 'mean')
    ).reset_index()

    section_avg_melted = section_avg.melt(
        id_vars='section',
        value_vars=['topic_0_count', 'topic_1_count'],
        var_name='Topic', value_name='Count'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Count', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section{suffix}')
    ax.set_ylabel('Average Count')
    ax.tick_params(axis='x', rotation=45)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['HUMAN', 'AI'], title='Topic')
    plt.tight_layout()
    plt.savefig(out_dir / f'average_ratio_count_selected{suffix}.png')
    plt.close()


def plot_simple_ratio(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        simple_ratio=('simple_ratio', 'mean')
    ).reset_index()
    section_avg_melted = section_avg.melt(
        id_vars='section', value_vars=['simple_ratio'],
        var_name='Topic', value_name='Ratio'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Ratio', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section{suffix}')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Average Ratio, 1 = HUMAN agency, 0 = AI use')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / f'average_ratio_simple_selected{suffix}.png')
    plt.close()


def plot_weighted_ratio(ratio_df: pd.DataFrame, out_dir: Path, suffix: str = ""):
    section_avg = ratio_df.groupby('section', observed=False).agg(
        weighted_ratio=('weighted_ratio', 'mean')
    ).reset_index()
    section_avg_melted = section_avg.melt(
        id_vars='section', value_vars=['weighted_ratio'],
        var_name='Topic', value_name='Ratio'
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=section_avg_melted, x='section', y='Ratio', hue='Topic', ax=ax)
    ax.set_title(f'Average Topic Distribution by Section, weighed by probability{suffix}')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Average Ratio, 1 = HUMAN agency, 0 = AI use')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.savefig(out_dir / f'average_ratio_weighed_selected{suffix}.png')
    plt.close()


def compute_section_similarities(df: pd.DataFrame, sections: list) -> pd.DataFrame:
    """Compute per-section average similarities and derived metrics."""
    sub = apply_section_order(df, sections)
    sims = sub.groupby('section', observed=False).agg(
        sim_to_HUMAN=('sim_to_HUMAN', 'mean'),
        sim_to_AI=('sim_to_AI', 'mean')
    ).reset_index()

    sims['sim_difference'] = sims['sim_to_HUMAN'] - sims['sim_to_AI']

    min_d, max_d = sims['sim_difference'].min(), sims['sim_difference'].max()
    denom = max_d - min_d
    sims['normalized_difference'] = (
        (sims['sim_difference'] - min_d) / denom if denom != 0 else 0.5
    )
    sims['balanced_ratio'] = (
        (sims['sim_to_HUMAN'] - sims['sim_to_AI']) /
        (sims['sim_to_HUMAN'] + sims['sim_to_AI'])
    )
    return sims


def plot_similarity_balance_normalized(section_sims: pd.DataFrame, out_dir: Path,
                                       suffix: str = "", filename: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    section_sims_melted = section_sims.melt(
        id_vars='section',
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    ax1 = axes[0]
    sns.barplot(data=section_sims_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 0.6)
    ax1.set_title(f'Average Cosine Similarity by Section{suffix}')

    ax1.set_ylabel('Average Similarity')
    ax1.tick_params(axis='x', rotation=45)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['HUMAN', 'AI'],
               title='Reference Vector', loc="upper right")

    ax2 = axes[1]
    neutral = 0.5
    bar_lengths = section_sims['normalized_difference'].values - neutral
    colors = [green if v > neutral else 'skyblue'
              for v in section_sims['normalized_difference']]
    ax2.barh(section_sims['section'], bar_lengths, left=neutral,
             color=colors, alpha=0.7)
    ax2.axvline(x=0.5, color='grey', linestyle='--', linewidth=2, label='Neutral')
    ax2.set_xlim(0.0, 1.0)
    ax2.set_title(f'Similarity Balance: 0 = AI, 1 = HUMAN{suffix}')
    ax2.set_ylabel('Section')
    ax2.set_xlabel('Normalized Difference')
    ax2.legend(loc='lower right')
    ax2.invert_yaxis()

    plt.tight_layout()
    fname = filename or f'similarity_balance_normalized{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def plot_boxplots_both_similarities(df: pd.DataFrame, sections: list,
                                    out_dir: Path, suffix: str = "", filename: str = ""):
    sub = apply_section_order(df, sections)
    sub['balanced_ratio'] = (
        (sub['sim_to_HUMAN'] - sub['sim_to_AI']) /
        (sub['sim_to_HUMAN'] + sub['sim_to_AI'])
    )

    triplets_melted = sub.melt(
        id_vars=['section', 'transcript_id'],
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.boxplot(data=triplets_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f'Distribution of Similarities by Section{suffix}')
    ax1.set_xlabel('Section')
    ax1.set_ylabel('Cosine Similarity')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['HUMAN', 'AI'],
               title='Reference Vector', loc='upper right')

    ax2 = axes[1]
    sns.boxplot(data=sub, x='section', y='balanced_ratio', ax=ax2, color=green)
    ax2.axhline(y=0, color='grey', linestyle=':', linewidth=2, label='Equal similarity')
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title(f'Balanced Similarity Ratio by Section\n(-1=AI, +1=HUMAN){suffix}')
    ax2.set_xlabel('Section')
    ax2.set_ylabel('Balanced Ratio')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    fname = filename or f'similarity_balanced_ratio{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def plot_normalized_difference_boxplots(df: pd.DataFrame, sections: list,
                                        out_dir: Path, suffix: str = "", filename: str = ""):
    sub = apply_section_order(df, sections).copy()
    sub['sim_difference_individual'] = sub['sim_to_HUMAN'] - sub['sim_to_AI']
    min_d, max_d = sub['sim_difference_individual'].min(), sub['sim_difference_individual'].max()
    denom = max_d - min_d
    sub['normalized_difference'] = (
        (sub['sim_difference_individual'] - min_d) / denom if denom != 0 else 0.5
    )

    triplets_melted = sub.melt(
        id_vars=['section', 'transcript_id'],
        value_vars=['sim_to_HUMAN', 'sim_to_AI'],
        var_name='Vector', value_name='Similarity'
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    sns.boxplot(data=triplets_melted, x='section', y='Similarity',
                hue='Vector', ax=ax1, palette="colorblind")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f'Distribution of Similarities by Section{suffix}')
    ax1.set_xlabel('Section')
    ax1.set_ylabel('Cosine Similarity')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=['HUMAN', 'AI'],
               title='Reference Vector', loc='upper right')

    ax2 = axes[1]
    sns.boxplot(data=sub, x='section', y='normalized_difference', ax=ax2, color=green)
    ax2.axhline(y=0.5, color='grey', linestyle=':', linewidth=2, label='Equal similarity')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title(f'Normalized Similarity Difference by Section\n(0=AI, 1=HUMAN){suffix}')
    ax2.set_xlabel('Section')
    ax2.set_ylabel('Normalized Difference')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    fname = filename or f'similarity_normalized_differences{suffix}.png'
    plt.savefig(out_dir / fname)
    plt.close()


def run_analysis_for_group(df: pd.DataFrame, out_dir: Path,
                           label: str = "", ratio_df: pd.DataFrame = None):
    """
    Run the full plotting + summary pipeline for one subset of all_triplets.
    df       : rows of all_triplets (already has sim_to_HUMAN, sim_to_AI, etc.)
    out_dir  : where to write outputs
    label    : display label for plot titles (e.g. '' or ' [writer]')
    ratio_df : pre-computed BERTopic ratio dataframe for the same subset
    """
    suffix = f" [{label}]" if label else ""

    # ── BERTopic ratio plots (only when ratio_df is provided) ─────────────────
    if ratio_df is not None:
        ratio_sub = apply_section_order(ratio_df, section_order_selected)
        plot_average_topic_counts(ratio_sub, out_dir, suffix)
        plot_simple_ratio(ratio_sub, out_dir, suffix)
        plot_weighted_ratio(ratio_sub, out_dir, suffix)

    # ── for ALL sections – not only first 4 relevant ────────────────────────
    section_sims_all = compute_section_similarities(df, section_order_all)
    # section_sims_all.to_csv(
    #     out_dir / f'{label}__average_similarities_ratio_HUMAN-AI_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.tsv',
    #     index=False
    # )
    # plot_similarity_balance_normalized(
    #     section_sims_all, out_dir, suffix,
    #     filename=f'similarity_balance_normalized_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    # )

    # boxplots – all sections
    # plot_boxplots_both_similarities(
    #     df, section_order_all, out_dir, suffix,
    #     filename=f'{label}__similarity_balanced_ratio_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    # )
    # plot_normalized_difference_boxplots(
    #     df, section_order_all, out_dir, suffix,
    #     filename=f'{label}__similarity_normalized_differences_all_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    # )

    # ── SELECTED sections ──────────────────────────────────────────────────────
    section_sims_sel = compute_section_similarities(df, section_order_selected)

    plot_similarity_balance_normalized(
        section_sims_sel, out_dir, suffix,
        filename=f'{label}__similarity_normalized_difference_start_sections{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )
    plot_boxplots_both_similarities(
        df, section_order_selected, out_dir, suffix,
        filename=f'{label}__similarity_balanced_difference_start_sections_all_docs{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )
    plot_normalized_difference_boxplots(
        df, section_order_selected, out_dir, suffix,
        filename=f'{label}__similarity_normalized_differences_four_sections_all_documents{suffix.replace(" ", "_").replace("[","").replace("]","")}.png'
    )

    # ── Summary printout ───────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"Summary by Section [{label or 'ALL'}]:")
    print(section_sims_all[['section', 'sim_to_HUMAN', 'sim_to_AI', 'normalized_difference']])



