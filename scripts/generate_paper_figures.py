#!/usr/bin/env python3
"""
Generate publication-quality figures for the Filo-Priori IEEE TSE paper.

Two key figures:
1. Cross-dataset APFD comparison (RQ1) — bar chart with error bars
2. Cross-dataset ablation (RQ2) — component contribution comparison

Uses Matplotlib + SciencePlots with IEEE style.
Exports as PDF (vector) for LaTeX inclusion.

Usage:
    python scripts/generate_paper_figures.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX (not installed in this env)
import numpy as np
import scienceplots
from pathlib import Path

OUTPUT_DIR = Path('paper/figures')
OUTPUT_DIR.mkdir(exist_ok=True)


def fig1_apfd_comparison():
    """
    Figure 1: Cross-Dataset APFD Comparison (RQ1)

    Grouped bar chart showing Filo-Priori vs 5 DL baselines on both datasets.
    Highlights the different competitive landscapes.
    """

    methods = ['Filo-Priori', 'DeepOrder', 'TCP-Net', 'NodeRank', 'RETECS', 'FailRank-BB']

    # Industrial dataset (277 builds)
    ind_apfd = [0.7611, 0.6890, 0.6704, 0.6609, 0.6406, 0.5953]
    ind_std  = [0.189,  0.266,  0.271,  0.270,  0.281,  0.263]

    # RTPTorrent (20 projects)
    rtp_apfd = [0.8540, 0.8136, 0.8253, 0.8038, 0.6791, 0.8218]
    rtp_std  = [0.112,  0.104,  0.110,  0.109,  0.156,  0.092]

    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=False)

        x = np.arange(len(methods))
        width = 0.65

        # Colors: Filo-Priori highlighted, baselines in grey scale
        colors_ind = ['#1565C0'] + ['#90A4AE'] * 5
        colors_rtp = ['#E65100'] + ['#BCAAA4'] * 5

        # --- Panel A: Industrial ---
        bars1 = ax1.bar(x, ind_apfd, width, yerr=ind_std, capsize=2,
                       color=colors_ind, edgecolor='black', linewidth=0.5,
                       error_kw={'linewidth': 0.7, 'capthick': 0.7})

        ax1.set_ylabel('APFD', fontsize=9)
        ax1.set_title('(a) Industrial QTA (277 builds)', fontsize=9, pad=8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=7)
        ax1.set_ylim(0.45, 0.95)
        ax1.axhline(y=ind_apfd[0], color='#1565C0', linestyle='--', linewidth=0.6, alpha=0.5)

        # Add significance markers
        ax1.text(0, ind_apfd[0] + ind_std[0] + 0.02, '$p<0.001$\nfor all',
                ha='center', va='bottom', fontsize=5.5, color='#1565C0', style='italic')

        # Add value labels on bars
        for i, (v, s) in enumerate(zip(ind_apfd, ind_std)):
            ax1.text(i, v + s + 0.01, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=5.5, fontweight='bold' if i == 0 else 'normal')

        # --- Panel B: RTPTorrent ---
        bars2 = ax2.bar(x, rtp_apfd, width, yerr=rtp_std, capsize=2,
                       color=colors_rtp, edgecolor='black', linewidth=0.5,
                       error_kw={'linewidth': 0.7, 'capthick': 0.7})

        ax2.set_ylabel('APFD', fontsize=9)
        ax2.set_title('(b) RTPTorrent (20 projects)', fontsize=9, pad=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=7)
        ax2.set_ylim(0.55, 1.0)
        ax2.axhline(y=rtp_apfd[0], color='#E65100', linestyle='--', linewidth=0.6, alpha=0.5)

        # Add value labels
        for i, (v, s) in enumerate(zip(rtp_apfd, rtp_std)):
            ax2.text(i, v + s + 0.01, f'{v:.3f}', ha='center', va='bottom',
                    fontsize=5.5, fontweight='bold' if i == 0 else 'normal')

        # Improvement annotations for industrial
        for i in range(1, len(methods)):
            delta = (ind_apfd[0] - ind_apfd[i]) / ind_apfd[i] * 100
            ax1.annotate(f'+{delta:.1f}%', xy=(i, ind_apfd[i]),
                        xytext=(i, ind_apfd[i] - 0.04),
                        ha='center', va='top', fontsize=5, color='#37474F')

        plt.tight_layout(w_pad=1.5)

        path = OUTPUT_DIR / 'fig_apfd_comparison.pdf'
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f'Saved: {path}')


def fig2_ablation_crossdataset():
    """
    Figure 2: Cross-Dataset Ablation Study (RQ2)

    Side-by-side horizontal bar charts showing component contributions
    on Industrial vs RTPTorrent, highlighting complementary patterns.
    """

    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

        # --- Industrial: Component Isolation (Table ablation_isolation) ---
        ind_components = [
            'GATv2\nGraph Attention',
            'Structural\nStream',
            'Class\nBalancing',
            'DNN\nEnsemble',
            'Semantic\nStream',
            'Cross-\nAttention'
        ]
        ind_contrib = [17.0, 5.3, 4.6, 3.5, 1.9, -1.1]
        ind_sig = ['***', '**', '*', '*', 'ns', 'ns']
        ind_colors = ['#1565C0' if c > 0 and s != 'ns' else '#90CAF9' if c > 0 else '#FFCDD2'
                     for c, s in zip(ind_contrib, ind_sig)]

        y_pos = np.arange(len(ind_components))
        bars1 = ax1.barh(y_pos, ind_contrib, height=0.6, color=ind_colors,
                        edgecolor='black', linewidth=0.5)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ind_components, fontsize=6.5)
        ax1.set_xlabel('Contribution to APFD (%)', fontsize=8)
        ax1.set_title('(a) Industrial Dataset', fontsize=9, pad=8)
        ax1.axvline(x=0, color='black', linewidth=0.5)
        ax1.set_xlim(-3, 20)
        ax1.invert_yaxis()

        # Add significance labels
        for i, (c, s) in enumerate(zip(ind_contrib, ind_sig)):
            offset = 0.5 if c >= 0 else -0.5
            ha = 'left' if c >= 0 else 'right'
            label = f'{c:+.1f}% ({s})'
            ax1.text(c + offset, i, label, ha=ha, va='center', fontsize=6,
                    fontweight='bold' if s not in ['ns'] else 'normal')

        # --- RTPTorrent: Ablation (Table ablation_rtptorrent) ---
        rtp_components = [
            'DNN\nEnsemble',
            'GATv2',
            'Semantic\nStream',
            'Co-failure\nGraph',
            'DeepOrder\nFeatures'
        ]
        rtp_contrib = [13.1, 0.0, 0.0, -0.0, 0.0]
        rtp_sig = ['***', 'ns', 'ns', 'ns', 'ns']
        rtp_colors = ['#E65100' if c > 0 and s != 'ns' else '#FFE0B2'
                     for c, s in zip(rtp_contrib, rtp_sig)]

        y_pos2 = np.arange(len(rtp_components))
        bars2 = ax2.barh(y_pos2, rtp_contrib, height=0.6, color=rtp_colors,
                        edgecolor='black', linewidth=0.5)

        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(rtp_components, fontsize=6.5)
        ax2.set_xlabel('Contribution to APFD (%)', fontsize=8)
        ax2.set_title('(b) RTPTorrent (20 projects)', fontsize=9, pad=8)
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.set_xlim(-2, 16)
        ax2.invert_yaxis()

        # Add significance labels
        for i, (c, s) in enumerate(zip(rtp_contrib, rtp_sig)):
            offset = 0.5
            label = f'{c:+.1f}% ({s})'
            ax2.text(max(c, 0) + offset, i, label, ha='left', va='center', fontsize=6,
                    fontweight='bold' if s not in ['ns'] else 'normal')

        # Add annotation highlighting the key finding
        fig.text(0.5, -0.06,
                'Graph-based modeling dominates with rich metadata (a); '
                'DNN ensemble dominates with sparse metadata (b)',
                ha='center', va='top', fontsize=7, style='italic',
                color='#37474F')

        plt.tight_layout(w_pad=2.0)

        path = OUTPUT_DIR / 'fig_ablation_crossdataset.pdf'
        plt.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        print(f'Saved: {path}')


if __name__ == '__main__':
    print('Generating publication figures...')
    print(f'Output directory: {OUTPUT_DIR}')
    fig1_apfd_comparison()
    fig2_ablation_crossdataset()
    print('Done.')
