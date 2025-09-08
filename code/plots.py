import numpy as np
import matplotlib.pyplot as plt
def plot_auc_vs_time(curves, labels, fig_path=None, title="AUC vs. Time"):
    plt.figure(figsize=(6,4))
    for df, lb in zip(curves, labels):
        plt.plot(df['T'], df['AUC'], label=lb, linewidth=2)
    plt.axhline(0.70, linestyle='--', alpha=0.5)
    plt.xlabel("Accumulated time (s)")
    plt.ylabel("AUC (LOSO)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    return plt.gcf()
