import matplotlib
matplotlib.use('Agg')

"""
Gather average similarity TSV files across creative type folders and plot comparisons.

Usage:
    python plot_similarities_by_type.py --results_dir results_new/manual_vectors --out plots/comparison
    python plot_similarities_by_type.py --results_dir results_new/manual_vectors --out plots/comparison --sections walkthrough project_example dynamic
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SECTION_ORDER = [
    'basic_job_description', 'walkthrough', 'project_example',
    'dynamic'
]
CREATIVE_TYPES = ['arts_crafts', 'design', 'film_video', 'music', 'photo', 'screenwriting', 'web_content', 'writing_fiction', 'writing_misc']
METRICS = ["sim_to_HUMAN","sim_to_AI","sim_difference","normalized_difference","balanced_ratio"]


def find_similarity_files(results_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Walk results_dir and collect all TSV files matching
    *average_similarities*.tsv pattern, keyed by creative type folder name.
    """
    collected = {}
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in CREATIVE_TYPES:
            continue
        label = folder.name
        for f in folder.glob("*average_similarities_ratio*.tsv"):
            try:
                df = pd.read_csv(f)
                df.columns = df.columns.str.strip()
                collected[label] = df
                print(f"  Loaded: {f}  ({len(df)} rows)")
                break  # take first match per folder
            except Exception as e:
                print(f"  WARNING: could not read {f}: {e}")

    return collected


def align_sections(dfs: dict[str, pd.DataFrame], sections: list[str]) -> dict[str, pd.DataFrame]:
    """Filter and reorder sections consistently across all dataframes."""
    aligned = {}
    for label, df in dfs.items():
        df = df[df['section'].isin(sections)].copy()
        df['section'] = pd.Categorical(df['section'], categories=sections, ordered=True)
        df = df.sort_values('section')
        aligned[label] = df
    return aligned


def build_long_df(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all creative type dataframes into one long-format dataframe."""
    rows = []
    for label, df in dfs.items():
        for _, row in df.iterrows():
            rows.append({
                'creative_type': label,
                'section': row['section'],
                'sim_to_HUMAN': row.get('avg_sim_to_a', row.get('sim_to_HUMAN', np.nan)),
                'sim_to_AI': row.get('avg_sim_to_b', row.get('sim_to_AI', np.nan)),
            })
    long = pd.DataFrame(rows)
    long['balanced_ratio'] = (
            (long['sim_to_HUMAN'] - long['sim_to_AI']) /
            (long['sim_to_HUMAN'] + long['sim_to_AI'])
    )
    long['sim_difference'] = long['sim_to_HUMAN'] - long['sim_to_AI']
    return long


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sim_by_section_and_type(long_df: pd.DataFrame, out_dir: Path):
    """Grouped bar: avg similarity to HUMAN and AI per section, one panel per creative type."""
    creative_types = sorted(long_df['creative_type'].unique())
    n = len(creative_types)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    palette = sns.color_palette('colorblind', 2)

    for ax, ct in zip(axes, creative_types):
        sub = long_df[long_df['creative_type'] == ct].copy()
        melted = sub.melt(
            id_vars='section',
            value_vars=['sim_to_HUMAN', 'sim_to_AI'],
            var_name='Vector', value_name='Similarity'
        )
        sns.barplot(data=melted, x='section', y='Similarity',
                    hue='Vector', ax=ax, palette=palette)
        ax.set_title(ct)
        ax.set_xlabel('Section')
        ax.set_ylabel('Avg Cosine Similarity' if ct == creative_types[0] else '')
        ax.tick_params(axis='x', rotation=45)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=['HUMAN', 'AI'], title='Vector')

    plt.suptitle('Average Similarity to HUMAN vs AI by Section and Creative Type', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'sim_by_section_and_type.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: sim_by_section_and_type.png")


def plot_balanced_ratio_comparison(long_df: pd.DataFrame, out_dir: Path):
    """Line plot: balanced ratio per section, one line per creative type."""
    fig, ax = plt.subplots(figsize=(11, 5))
    palette = sns.color_palette('colorblind', long_df['creative_type'].nunique())

    for i, (ct, sub) in enumerate(long_df.groupby('creative_type')):
        sub = sub.sort_values('section')
        ax.plot(sub['section'], sub['balanced_ratio'],
                marker='o', label=ct, color=palette[i], linewidth=2)

    ax.axhline(0, color='grey', linestyle='--', linewidth=1, label='Equal similarity')
    ax.set_title('Balanced Similarity Ratio by Section and Creative Type\n(+1 = HUMAN, -1 = AI)')
    ax.set_xlabel('Section')
    ax.set_ylabel('Balanced Ratio')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Creative Type', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / 'balanced_ratio_by_type.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: balanced_ratio_by_type.png")


def plot_heatmap_balanced_ratio(long_df: pd.DataFrame, out_dir: Path):
    """Heatmap: creative type x section, value = balanced ratio."""
    pivot = long_df.pivot_table(
        index='creative_type', columns='section', values='balanced_ratio'
    )
    # Reorder columns to section order
    cols = [s for s in SECTION_ORDER if s in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.4 + 1, len(pivot) * 0.8 + 1))
    sns.heatmap(
        pivot, ax=ax, cmap='RdYlGn', center=0,
        vmin=-0.3, vmax=0.3,
        annot=True, fmt='.2f', linewidths=0.5,
        cbar_kws={'label': 'Balanced Ratio (+= HUMAN, -= AI)'}
    )
    ax.set_title('Balanced Similarity Ratio: Creative Type × Section')
    ax.set_xlabel('Section')
    ax.set_ylabel('Creative Type')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / 'heatmap_balanced_ratio.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: heatmap_balanced_ratio.png")


def plot_sim_difference_boxplot(long_df: pd.DataFrame, out_dir: Path):
    """Boxplot: sim difference (HUMAN - AI) per creative type, faceted by section."""
    fig, ax = plt.subplots(figsize=(12, 5))
    palette = sns.color_palette('colorblind', long_df['creative_type'].nunique())
    sns.boxplot(
        data=long_df, x='section', y='sim_difference',
        hue='creative_type', ax=ax, palette=palette
    )
    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title('Similarity Difference (HUMAN − AI) by Section and Creative Type')
    ax.set_xlabel('Section')
    ax.set_ylabel('Sim(HUMAN) − Sim(AI)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Creative Type', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / 'sim_difference_by_type.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: sim_difference_by_type.png")


def save_combined_tsv(long_df: pd.DataFrame, out_dir: Path):
    out_path = out_dir / 'combined_similarities_all_types.tsv'
    long_df.to_csv(out_path, sep='\t', index=False)
    print(f"  Saved combined TSV: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare average similarities across creative types")
    ap.add_argument("--results_dir", required=True,
                    help="Root results directory containing one subfolder per creative type")
    ap.add_argument("--out", required=True,
                    help="Output directory for plots and combined TSV")
    ap.add_argument("--sections", nargs='+', default=SECTION_ORDER,
                    help="Sections to include (default: only 4)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nScanning: {results_dir}")
    dfs = find_similarity_files(results_dir)

    if not dfs:
        print("ERROR: No similarity files found. Check --results_dir and file naming.")
        exit(1)

    print(f"\nFound {len(dfs)} creative type(s): {sorted(dfs.keys())}")

    dfs = align_sections(dfs, args.sections)
    long_df = build_long_df(dfs)

    # Apply section ordering
    long_df['section'] = pd.Categorical(long_df['section'], categories=args.sections, ordered=True)
    long_df = long_df.sort_values(['creative_type', 'section'])

    print(f"\nGenerating plots in: {out_dir}")
    plot_sim_by_section_and_type(long_df, out_dir)
    plot_balanced_ratio_comparison(long_df, out_dir)
    plot_heatmap_balanced_ratio(long_df, out_dir)
    plot_sim_difference_boxplot(long_df, out_dir)
    save_combined_tsv(long_df, out_dir)

    print(f"\nOutputs written to: {out_dir}/")