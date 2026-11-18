from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def find_latest_results_json(search_dirs: list[Path] | None = None) -> Path:
    """
    Locate the most recent 'results_*.json' file across candidate directories.
    Prioritizes timestamp ordering in the filename, falling back to file modification time.
    """
    if search_dirs is None:
        script_dir = Path(__file__).resolve().parent
        cwd = Path.cwd().resolve()
        search_dirs = [cwd] if cwd == script_dir else [cwd, script_dir]

    candidates: dict[Path, Path] = {}
    for directory in search_dirs:
        if directory.exists() and directory.is_dir():
            for path in directory.glob("results_*.json"):
                candidates[path.resolve()] = path.resolve()

    if not candidates:
        searched = ", ".join(str(d) for d in search_dirs)
        raise FileNotFoundError(
            f"No 'results_*.json' file found in [{searched}]. "
            "Pass a specific file path using --input /path/to/results.json."
        )

    timestamp_pattern = re.compile(r"results_(\d{8}_\d{6})\.json$")

    def _sort_key(p: Path) -> tuple[int, str, float]:
        match = timestamp_pattern.search(p.name)
        ts_str = match.group(1) if match else ""
        return (1 if match else 0, ts_str, p.stat().st_mtime)

    return max(candidates.values(), key=_sort_key)


def load_results(json_path: Path) -> dict:
    """Load and validate the experiment JSON structure."""
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"seeds", "summary", "per_seed"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Invalid results JSON format. Missing keys: {sorted(missing)}")

    return data


def build_dataframes(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw JSON dictionary into structured summary and per-seed DataFrames."""
    mode_labels = {
        "random": "vs Random Agent",
        "identical": "Self-Play (Identical)",
    }

    summary_rows = []
    for mode_key, label in mode_labels.items():
        if mode_key not in data["summary"]:
            continue
        s = data["summary"][mode_key]
        summary_rows.append({
            "mode_key": mode_key,
            "Condition": label,
            "Seeds_Count": len(data["seeds"]),
            "Matches_Mean": s["n"]["mean"],
            "Matches_Std": s["n"]["std"],
            "Win_A_Mean": s["%_A"]["mean"],
            "Win_A_Std": s["%_A"]["std"],
            "Win_B_Mean": s["%_B"]["mean"],
            "Win_B_Std": s["%_B"]["std"],
            "Draw_Mean": s["%_draw"]["mean"],
            "Draw_Std": s["%_draw"]["std"],
            "Reward_A_Mean": s["rew_A"]["mean"],
            "Reward_A_Std": s["rew_A"]["std"],
            "Reward_B_Mean": s["rew_B"]["mean"],
            "Reward_B_Std": s["rew_B"]["std"],
        })
    summary_df = pd.DataFrame(summary_rows)

    per_seed_rows = []
    for entry in data["per_seed"]:
        seed = entry["seed"]
        for mode_key, label in mode_labels.items():
            if mode_key not in entry:
                continue
            m = entry[mode_key]
            per_seed_rows.append({
                "Seed": int(seed),
                "mode_key": mode_key,
                "Condition": label,
                "Matches": int(m["n"]),
                "Win_A_Pct": float(m["%_A"]),
                "Win_B_Pct": float(m["%_B"]),
                "Draw_Pct": float(m["%_draw"]),
                "Reward_A": int(m["rew_A"]),
                "Reward_B": int(m["rew_B"]),
            })
    per_seed_df = pd.DataFrame(per_seed_rows)

    return summary_df, per_seed_df


def export_csv_tables(summary_df: pd.DataFrame, per_seed_df: pd.DataFrame, output_dir: Path) -> None:
    """Export summary and per-seed tables to CSV files."""
    summary_export = summary_df.drop(columns=["mode_key"])
    per_seed_export = per_seed_df.drop(columns=["mode_key"])

    summary_export.to_csv(output_dir / "summary_metrics.csv", index=False, float_format="%.4f")
    per_seed_export.to_csv(output_dir / "per_seed_metrics.csv", index=False, float_format="%.4f")


def export_latex_tables(summary_df: pd.DataFrame, per_seed_df: pd.DataFrame, output_dir: Path) -> None:
    """Export publication-ready LaTeX tables (using booktabs) with Mean +/- Std formatting."""
    # 1. Aggregated Summary LaTeX Table
    summary_lines = [
        "% Requires \\usepackage{booktabs} in your LaTeX preamble",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Aggregated Multi-Seed Performance Comparison (Mean $\\pm$ Std across seeds)}",
        "    \\label{tab:marl_summary_comparison}",
        "    \\begin{tabular}{lccccc}",
        "        \\toprule",
        "        \\textbf{Condition} & \\textbf{Matches ($n$)} & \\textbf{Win A (\\%)} & \\textbf{Win B (\\%)} & \\textbf{Draw (\\%)} & \\textbf{Net Reward A} \\\\",
        "        \\midrule",
    ]
    for _, row in summary_df.iterrows():
        n_str = f"{row['Matches_Mean']:.1f} $\\pm$ {row['Matches_Std']:.1f}"
        wa_str = f"{row['Win_A_Mean']:.2f} $\\pm$ {row['Win_A_Std']:.2f}"
        wb_str = f"{row['Win_B_Mean']:.2f} $\\pm$ {row['Win_B_Std']:.2f}"
        dr_str = f"{row['Draw_Mean']:.2f} $\\pm$ {row['Draw_Std']:.2f}"
        rw_str = f"{row['Reward_A_Mean']:.1f} $\\pm$ {row['Reward_A_Std']:.1f}"
        summary_lines.append(
            f"        {row['Condition']} & {n_str} & {wa_str} & {wb_str} & {dr_str} & {rw_str} \\\\"
        )
    summary_lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
        "",
    ])
    (output_dir / "summary_metrics.tex").write_text("\n".join(summary_lines), encoding="utf-8")

    # 2. Per-Seed Breakdown LaTeX Table
    seed_lines = [
        "% Requires \\usepackage{booktabs} in your LaTeX preamble",
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Per-Seed Breakdown of Match Outcomes and Net Rewards}",
        "    \\label{tab:marl_per_seed}",
        "    \\begin{tabular}{clccccc}",
        "        \\toprule",
        "        \\textbf{Seed} & \\textbf{Condition} & \\textbf{Matches} & \\textbf{Win A (\\%)} & \\textbf{Win B (\\%)} & \\textbf{Draw (\\%)} & \\textbf{Reward A} \\\\",
        "        \\midrule",
    ]
    sorted_df = per_seed_df.sort_values(by=["Seed", "mode_key"], ascending=[True, False])
    prev_seed = None
    for _, row in sorted_df.iterrows():
        if prev_seed is not None and row["Seed"] != prev_seed:
            seed_lines.append("        \\midrule")
        prev_seed = row["Seed"]
        seed_lines.append(
            f"        {int(row['Seed'])} & {row['Condition']} & {int(row['Matches'])} & "
            f"{row['Win_A_Pct']:.2f} & {row['Win_B_Pct']:.2f} & {row['Draw_Pct']:.2f} & {int(row['Reward_A'])} \\\\"
        )
    seed_lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
        "",
    ])
    (output_dir / "per_seed_metrics.tex").write_text("\n".join(seed_lines), encoding="utf-8")


def generate_plots(summary_df: pd.DataFrame, per_seed_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate four standalone high-resolution figures (300 DPI):
      1. 01_outcome_rates_variance.png   : Grouped barplot by condition (Win A, Win B, Draw side-by-side) + seed dots.
      2. 02_net_reward_distribution.png  : Boxplot + strip points + Mean +/- 1 SD of Net Reward A.
      3. 03_self_play_seed_dynamics.png  : Paired dumbbell plot showing Win A vs Win B across seeds in Self-Play.
      4. 04_cross_seed_volatility.png    : Grouped barplot by condition comparing cross-seed SD (Win A, Win B, Draw).
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    condition_palette = {
        "vs Random Agent": "#2b5c8f",
        "Self-Play (Identical)": "#d95f02",
    }
    metrics = [
        ("Win_A", "Win A (%)", "#2b5c8f"),
        ("Win_B", "Win B (%)", "#d95f02"),
        ("Draw", "Draw (%)", "#1b9e77"),
    ]
    conditions = summary_df["Condition"].tolist()
    x_cond = np.arange(len(conditions))
    bar_width = 0.25
    offsets = [-bar_width, 0.0, bar_width]
    rng_jitter = np.random.default_rng(42)

    # =========================================================================
    # FIGURE 1: Outcome Rates & Cross-Seed Dispersion (Grouped by Condition)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8.6, 5.6))
    max_y1 = 0.0

    for m_idx, (m_prefix, m_label, m_color) in enumerate(metrics):
        means = [
            float(summary_df[summary_df["Condition"] == cond].iloc[0][f"{m_prefix}_Mean"])
            for cond in conditions
        ]
        stds = [
            float(summary_df[summary_df["Condition"] == cond].iloc[0][f"{m_prefix}_Std"])
            for cond in conditions
        ]
        pos = x_cond + offsets[m_idx]

        bars = ax1.bar(
            pos,
            means,
            width=bar_width,
            yerr=stds,
            capsize=5,
            color=m_color,
            alpha=0.84,
            edgecolor="#222222",
            linewidth=0.85,
            error_kw={"elinewidth": 1.5, "ecolor": "#111111", "capthick": 1.3},
            label=f"{m_label} (Mean ± 1 SD)",
        )

        # Overlay individual seed dots for each condition
        for c_idx, cond in enumerate(conditions):
            y_vals = per_seed_df[per_seed_df["Condition"] == cond][f"{m_prefix}_Pct"].to_numpy()
            jitter = rng_jitter.uniform(-bar_width * 0.22, bar_width * 0.22, size=len(y_vals))
            ax1.scatter(
                np.full_like(y_vals, pos[c_idx], dtype=float) + jitter,
                y_vals,
                color="#ffffff",
                edgecolor="#111111",
                linewidth=1.1,
                s=38,
                zorder=5,
                label="Individual Seed" if (m_idx == 0 and c_idx == 0) else None,
            )
            if len(y_vals) > 0:
                max_y1 = max(max_y1, float(np.max(y_vals)))

        # Annotate mean +/- std above each bar
        for bar, mean_v, std_v in zip(bars, means, stds):
            top_y = mean_v + std_v + 1.8
            max_y1 = max(max_y1, top_y + 6.0)
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                top_y,
                f"{mean_v:.1f}%\n(±{std_v:.1f})",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#222222",
            )

    ax1.set_title("Outcome Rates & Cross-Seed Dispersion by Experimental Condition", fontsize=12.5, fontweight="bold", pad=14)
    ax1.set_xticks(x_cond)
    ax1.set_xticklabels(conditions, fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Percentage of Completed Matches (%)", fontsize=11)
    ax1.set_ylim(0, max(96.0, max_y1))
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    fig1.tight_layout()
    fig1.savefig(output_dir / "01_outcome_rates_variance.png", dpi=300)
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2: Net Cumulative Reward Distribution (Player A)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(7.6, 5.4))
    box_data = [
        per_seed_df[per_seed_df["Condition"] == cond]["Reward_A"].to_numpy()
        for cond in conditions
    ]
    bp = ax2.boxplot(
        box_data,
        positions=x_cond,
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#333333", "linewidth": 1.2},
        capprops={"color": "#333333", "linewidth": 1.2},
    )
    for patch, cond in zip(bp["boxes"], conditions):
        patch.set_facecolor(condition_palette[cond])
        patch.set_alpha(0.38)
        patch.set_edgecolor("#222222")

    for c_idx, cond in enumerate(conditions):
        y_rewards = per_seed_df[per_seed_df["Condition"] == cond]["Reward_A"].to_numpy()
        jitter = rng_jitter.uniform(-0.08, 0.08, size=len(y_rewards))
        ax2.scatter(
            np.full_like(y_rewards, c_idx, dtype=float) + jitter,
            y_rewards,
            color="#111111",
            s=46,
            zorder=5,
            label="Seed Outcome" if c_idx == 0 else None,
        )
        sub_sum = summary_df[summary_df["Condition"] == cond].iloc[0]
        ax2.errorbar(
            c_idx,
            sub_sum["Reward_A_Mean"],
            yerr=sub_sum["Reward_A_Std"],
            fmt="D",
            color="#b2182b",
            ecolor="#b2182b",
            elinewidth=1.8,
            capsize=6,
            markersize=7.5,
            zorder=6,
            label="Mean ± 1 SD" if c_idx == 0 else None,
        )

    ax2.axhline(0, color="#555555", linestyle="--", linewidth=1.2, label="Zero-Sum Symmetry (0)")
    ax2.set_xticks(x_cond)
    ax2.set_xticklabels(conditions, fontsize=11.5, fontweight="bold")
    ax2.set_title("Net Cumulative Reward Distribution (Player A)", fontsize=12.5, fontweight="bold", pad=14)
    ax2.set_ylabel("Net Reward (Wins A - Wins B)", fontsize=11)
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)

    fig2.tight_layout()
    fig2.savefig(output_dir / "02_net_reward_distribution.png", dpi=300)
    plt.close(fig2)

    # =========================================================================
    # FIGURE 3: Paired Dumbbell Plot for Self-Play Symmetry & Path Dependence
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8.2, 5.2))
    ident_df = per_seed_df[per_seed_df["mode_key"] == "identical"].sort_values("Seed").reset_index(drop=True)
    ident_sum = summary_df[summary_df["mode_key"] == "identical"].iloc[0]
    y_pos = np.arange(len(ident_df))

    ax3.axvspan(
        ident_sum["Win_A_Mean"] - ident_sum["Win_A_Std"],
        ident_sum["Win_A_Mean"] + ident_sum["Win_A_Std"],
        color="#2b5c8f",
        alpha=0.10,
        label="Win A (±1 SD band)",
    )
    ax3.axvspan(
        ident_sum["Win_B_Mean"] - ident_sum["Win_B_Std"],
        ident_sum["Win_B_Mean"] + ident_sum["Win_B_Std"],
        color="#d95f02",
        alpha=0.10,
        label="Win B (±1 SD band)",
    )
    ax3.axvline(ident_sum["Win_A_Mean"], color="#2b5c8f", linestyle=":", linewidth=1.4)
    ax3.axvline(ident_sum["Win_B_Mean"], color="#d95f02", linestyle="--", linewidth=1.4)

    for idx, row in ident_df.iterrows():
        ax3.plot(
            [row["Win_A_Pct"], row["Win_B_Pct"]],
            [idx, idx],
            color="#666666",
            linewidth=1.9,
            zorder=2,
        )

    ax3.scatter(
        ident_df["Win_A_Pct"],
        y_pos,
        color="#2b5c8f",
        s=90,
        zorder=4,
        label=f"Player A (Mean: {ident_sum['Win_A_Mean']:.1f}%)",
    )
    ax3.scatter(
        ident_df["Win_B_Pct"],
        y_pos,
        color="#d95f02",
        marker="s",
        s=85,
        zorder=4,
        label=f"Player B (Mean: {ident_sum['Win_B_Mean']:.1f}%)",
    )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f"Seed {int(s)}" for s in ident_df["Seed"]], fontsize=11)
    ax3.invert_yaxis()
    ax3.set_xlabel("Win Rate (%)", fontsize=11)
    ax3.set_title("Self-Play Symmetry & Path Dependence by Seed", fontsize=12.5, fontweight="bold", pad=14)
    ax3.set_xlim(0, max(ident_df[["Win_A_Pct", "Win_B_Pct"]].max()) + 6.5)
    ax3.legend(loc="lower right", frameon=True, fontsize=9.2)

    fig3.tight_layout()
    fig3.savefig(output_dir / "03_self_play_seed_dynamics.png", dpi=300)
    plt.close(fig3)

    # =========================================================================
    # FIGURE 4: Cross-Seed Standard Deviation (Grouped by Condition)
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(8.4, 5.4))
    max_std_val = 0.0

    for m_idx, (m_prefix, m_label, m_color) in enumerate(metrics):
        std_vals = [
            float(summary_df[summary_df["Condition"] == cond].iloc[0][f"{m_prefix}_Std"])
            for cond in conditions
        ]
        pos = x_cond + offsets[m_idx]
        bars_std = ax4.bar(
            pos,
            std_vals,
            width=bar_width,
            color=m_color,
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.85,
            label=m_label,
        )
        for bar, val in zip(bars_std, std_vals):
            max_std_val = max(max_std_val, val)
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.16,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#222222",
            )

    ax4.set_xticks(x_cond)
    ax4.set_xticklabels(conditions, fontsize=11.5, fontweight="bold")
    ax4.set_title("Instability Gap: Cross-Seed Standard Deviation (σ) by Condition", fontsize=12.5, fontweight="bold", pad=14)
    ax4.set_ylabel("Standard Deviation across Seeds (%)", fontsize=11)
    ax4.set_ylim(0, max_std_val * 1.24 if max_std_val > 0 else 1.0)
    ax4.legend(loc="upper left", frameon=True, fontsize=9.8)

    fig4.tight_layout()
    fig4.savefig(output_dir / "04_cross_seed_volatility.png", dpi=300)
    plt.close(fig4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CSV tables, LaTeX tables, and variance-focused plots from MARL JSON results."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Path to a specific results JSON file. Defaults to the latest 'results_*.json' found.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where tables and plots will be saved. Defaults to 'intel_reports/<json_stem>/'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = args.input.resolve() if args.input is not None else find_latest_results_json()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path.cwd() / "intel_reports" / json_path.stem).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from : {json_path}")
    print(f"Saving artifacts to  : {output_dir}")

    data = load_results(json_path)
    summary_df, per_seed_df = build_dataframes(data)

    export_csv_tables(summary_df, per_seed_df, output_dir)
    export_latex_tables(summary_df, per_seed_df, output_dir)
    generate_plots(summary_df, per_seed_df, output_dir)

    print("Artifacts generated successfully:")
    for file_path in sorted(output_dir.iterdir()):
        print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()