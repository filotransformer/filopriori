#!/usr/bin/env python3
"""
Generate publication-quality figures for the Filo-Priori IEEE TSE paper.
All data from the final validated results (CLAUDE.md / paper tables).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# IEEE TSE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def generate_apfd_comparison():
    """Figure 1: Cross-dataset APFD comparison (bar chart)."""

    methods = ['Filo-Priori', 'DeepOrder', 'TCP-Net', 'NodeRank', 'RETECS', 'FailRank-BB']

    # Industrial dataset (277 builds) - from Table tab:tcp_comparison
    ind_apfd = [0.761, 0.689, 0.670, 0.661, 0.641, 0.595]
    ind_std  = [0.189, 0.266, 0.271, 0.270, 0.281, 0.263]
    ind_delta = ['', '+10.2%', '+13.3%', '+14.9%', '+18.6%', '+27.6%']

    # RTPTorrent (20 projects) - from Table tab:rtptorrent_dl
    rtp_apfd = [0.854, 0.814, 0.825, 0.804, 0.679, 0.822]
    rtp_std  = [0.111, 0.104, 0.110, 0.109, 0.156, 0.092]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    x = np.arange(len(methods))
    bar_width = 0.6

    # --- Panel (a): Industrial ---
    colors_ind = ['#2563EB'] + ['#94A3B8'] * 5
    bars1 = ax1.bar(x, ind_apfd, bar_width, color=colors_ind, edgecolor='#334155',
                    linewidth=0.5, zorder=3)
    # Error bars (separate for cleaner look)
    ind_yerr = [np.zeros(len(ind_std)), ind_std]
    ax1.errorbar(x, ind_apfd, yerr=ind_yerr, fmt='none', ecolor='#1E293B',
                 elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    # APFD value labels above error bars
    for i, (v, s) in enumerate(zip(ind_apfd, ind_std)):
        ax1.text(i, v + s + 0.015, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold' if i == 0 else 'normal')

    ax1.set_title('(a) Industrial QTA (277 builds)', fontweight='bold', pad=8)
    ax1.set_ylabel('APFD')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=7.5)
    ax1.set_ylim(0.45, 1.05)
    ax1.axhline(y=ind_apfd[0], color='#2563EB', linestyle='--', linewidth=0.6, alpha=0.4)
    ax1.text(5.3, ind_apfd[0] + 0.008, 'p < 0.001\nfor all', fontsize=6.5,
             color='#2563EB', ha='right', va='bottom', style='italic')
    ax1.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax1.set_axisbelow(True)

    # --- Panel (b): RTPTorrent ---
    colors_rtp = ['#EA580C'] + ['#D4C5B2'] * 5
    bars2 = ax2.bar(x, rtp_apfd, bar_width, color=colors_rtp, edgecolor='#334155',
                    linewidth=0.5, zorder=3)
    rtp_yerr = [np.zeros(len(rtp_std)), rtp_std]
    ax2.errorbar(x, rtp_apfd, yerr=rtp_yerr, fmt='none', ecolor='#1E293B',
                 elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)

    for i, (v, s) in enumerate(zip(rtp_apfd, rtp_std)):
        ax2.text(i, v + s + 0.015, f'{v:.3f}', ha='center', va='bottom',
                 fontsize=7.5, fontweight='bold' if i == 0 else 'normal')

    ax2.set_title('(b) RTPTorrent (20 projects)', fontweight='bold', pad=8)
    ax2.set_ylabel('APFD')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=7.5)
    ax2.set_ylim(0.45, 1.10)
    ax2.axhline(y=rtp_apfd[0], color='#EA580C', linestyle='--', linewidth=0.6, alpha=0.4)
    ax2.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout(w_pad=2.0)
    plt.savefig('paper/figures/fig_apfd_comparison.pdf', format='pdf')
    plt.savefig('paper/figures/fig_apfd_comparison.png', format='png')
    print("Generated: fig_apfd_comparison.pdf")
    plt.close()


def generate_ablation_crossdataset():
    """Figure 2: Cross-dataset ablation (horizontal bar chart)."""

    # --- Industrial Component Isolation (Table tab:ablation_isolation) ---
    # Base Architecture = 0.6397; contribution = % decrease when removed
    ind_components = [
        'GATv2\nGraph Attention',
        'Structural\nStream',
        'Class\nBalancing',
        'DNN\nEnsemble',
        'Semantic\nStream',
        'Cross-\nAttention',
    ]
    ind_contributions = [17.0, 5.3, 4.6, 3.5, 1.9, -1.1]
    ind_significance  = ['***', '**', '*', '*', 'ns', 'ns']
    ind_colors = ['#1D4ED8', '#2563EB', '#3B82F6', '#60A5FA', '#BFDBFE', '#FECACA']

    # --- RTPTorrent Ablation (Table tab:ablation_rtptorrent) ---
    # Full Model = 0.8540; contribution = % decrease when removed
    rtp_components = [
        'DNN\nEnsemble',
        'Semantic\nStream',
        'GATv2',
        'Multi-Edge\nGraph',
    ]
    rtp_contributions = [2.6, 1.1, 1.0, 0.3]
    rtp_significance  = ['***', 'ns', '*', 'ns']
    rtp_colors = ['#EA580C', '#FED7AA', '#FB923C', '#FED7AA']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    # --- Panel (a): Industrial ---
    y_ind = np.arange(len(ind_components))
    bars1 = ax1.barh(y_ind, ind_contributions, height=0.6, color=ind_colors,
                     edgecolor='#334155', linewidth=0.5, zorder=3)

    for i, (v, sig) in enumerate(zip(ind_contributions, ind_significance)):
        if v >= 0:
            ax1.text(v + 0.3, i, f'+{v:.1f}% ({sig})', va='center', fontsize=7.5,
                     fontweight='bold' if sig == '***' else 'normal')
        else:
            # Negative values: place label to the right of zero
            ax1.text(0.3, i, f'{v:.1f}% ({sig})', va='center', ha='left',
                     fontsize=7.5)

    ax1.set_yticks(y_ind)
    ax1.set_yticklabels(ind_components, fontsize=7.5)
    ax1.set_ylabel('Removed Component')
    ax1.set_xlabel('Contribution to APFD (%)')
    ax1.set_title('(a) Industrial Dataset', fontweight='bold', pad=8)
    ax1.set_xlim(-3, 22)
    ax1.axvline(x=0, color='#64748B', linewidth=0.5)
    ax1.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.invert_yaxis()

    # --- Panel (b): RTPTorrent ---
    y_rtp = np.arange(len(rtp_components))
    bars2 = ax2.barh(y_rtp, rtp_contributions, height=0.6, color=rtp_colors,
                     edgecolor='#334155', linewidth=0.5, zorder=3)

    for i, (v, sig) in enumerate(zip(rtp_contributions, rtp_significance)):
        label = f'+{v:.1f}% ({sig})' if v > 0 else f'{v:.1f}% ({sig})'
        ax2.text(v + 0.1, i, label, va='center', fontsize=7.5,
                 fontweight='bold' if sig == '***' else 'normal')

    ax2.set_yticks(y_rtp)
    ax2.set_yticklabels(rtp_components, fontsize=7.5)
    ax2.set_ylabel('Removed Component')
    ax2.set_xlabel('Contribution to APFD (%)')
    ax2.set_title('(b) RTPTorrent (20 projects)', fontweight='bold', pad=8)
    ax2.set_xlim(-0.5, 5)
    ax2.axvline(x=0, color='#64748B', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.2, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()

    plt.tight_layout(w_pad=2.0)
    plt.savefig('paper/figures/fig_ablation_crossdataset.pdf', format='pdf')
    plt.savefig('paper/figures/fig_ablation_crossdataset.png', format='png')
    print("Generated: fig_ablation_crossdataset.pdf")
    plt.close()


if __name__ == '__main__':
    generate_apfd_comparison()
    generate_ablation_crossdataset()
    print("All figures generated successfully.")
