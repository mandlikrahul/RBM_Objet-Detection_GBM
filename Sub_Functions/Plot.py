"""
Plot.py
=======
Graph plotting for comparative and performance results.
Reads CSVs produced by Analysis.py and generates publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = sns.color_palette("tab10")


class ALL_GRAPH_PLOT:
    """Plot all result graphs for the given dataset."""

    def GRAPH_RESULT(self, db_name: str):
        print(f"\n[Plot] Generating graphs for {db_name} …")

        comp_path = os.path.join(RESULTS_DIR, f"{db_name}_comparative.csv")
        perf_path = os.path.join(RESULTS_DIR, f"{db_name}_performance.csv")

        if not os.path.exists(comp_path):
            print(f"[Plot] {comp_path} not found — run Analysis first.")
            return
        if not os.path.exists(perf_path):
            print(f"[Plot] {perf_path} not found — run Analysis first.")
            return

        comp_df = pd.read_csv(comp_path)
        perf_df = pd.read_csv(perf_path)

        self._bar_comparison(comp_df, db_name, metric="mAP@0.5",  ylabel="mAP@0.5 (%)")
        self._bar_comparison(comp_df, db_name, metric="Precision", ylabel="Precision (%)")
        self._bar_comparison(comp_df, db_name, metric="Recall",    ylabel="Recall (%)")
        self._bar_comparison(comp_df, db_name, metric="F1-Score",  ylabel="F1-Score (%)")
        self._multi_metric_radar(comp_df, db_name)
        self._fps_plot(comp_df, db_name)
        self._perf_bar(perf_df, db_name)

        print(f"[Plot] All graphs saved to {PLOTS_DIR}/")

    # ── Bar chart for one metric ──────────────────────────────────────────────
    def _bar_comparison(self, df, db_name, metric, ylabel):
        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        bars    = ax.bar(df["Method"], df[metric], color=colors, edgecolor="black", width=0.6)

        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Comparison — {db_name}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, df[metric].max() * 1.15)
        plt.xticks(rotation=20, ha="right", fontsize=10)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{db_name}_{metric.replace('@','_')}_bar.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Radar / spider chart ──────────────────────────────────────────────────
    def _multi_metric_radar(self, df, db_name):
        metrics  = ["mAP@0.5", "mAP@0.75", "Precision", "Recall", "F1-Score"]
        n        = len(metrics)
        angles   = [i / float(n) * 2 * np.pi for i in range(n)]
        angles  += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for idx, row in df.iterrows():
            values  = [row[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row["Method"],
                    color=PALETTE[idx % len(PALETTE)])
            ax.fill(angles, values, alpha=0.08, color=PALETTE[idx % len(PALETTE)])

        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)
        ax.set_title(f"Multi-Metric Radar — {db_name}", fontsize=13, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{db_name}_radar.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── FPS comparison ────────────────────────────────────────────────────────
    def _fps_plot(self, df, db_name):
        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
        bars    = ax.barh(df["Method"], df["FPS"], color=colors, edgecolor="black")

        for bar, val in zip(bars, df["FPS"]):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("Frames per Second (FPS)", fontsize=12)
        ax.set_title(f"Inference Speed Comparison — {db_name}", fontsize=13, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{db_name}_FPS_bar.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Performance metrics bar for the proposed model only ───────────────────
    def _perf_bar(self, df, db_name):
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values  = [df.iloc[0][m] for m in metrics if m in df.columns]
        labels  = [m for m in metrics if m in df.columns]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars    = ax.bar(labels, values, color=PALETTE[:len(labels)], edgecolor="black", width=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylim(0, 110)
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title(f"Proposed Model Performance — {db_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = os.path.join(PLOTS_DIR, f"{db_name}_performance_bar.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
