import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

EVAL_DIR   = "/kaggle/working/ukiyo-lora/eval"
OUTPUT_DIR = "/kaggle/working/ukiyo-lora/output"

FID_CSV  = os.path.join(EVAL_DIR, "fid_scores.csv")
CLIP_CSV = os.path.join(EVAL_DIR, "clip_scores.csv")

RUNS = [
    {"key": "run_4",  "name": "run_4_1e-4",  "csv_key": "run_4",  "rank": 4,  "lr": "1e-4",
     "loss_log": os.path.join(OUTPUT_DIR, "run_4_1e-4",  "loss_log.csv")},
    {"key": "run_32", "name": "run_32_1e-04", "csv_key": "run_32", "rank": 32, "lr": "1e-4",
     "loss_log": os.path.join(OUTPUT_DIR, "run_32_1e-04", "loss_log.csv")},
]

SAMPLE_STEPS = [500, 1000, 1500, 2000]


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_metric_csv(path, value_col_hint):
    """Return {run_name: float} from a two-column CSV (run, <metric>)."""
    result = {}
    if not os.path.exists(path):
        print(f"  [warn] not found: {path}")
        return result
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # normalise header keys
        fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        run_col  = next((h for h in fieldnames if "run"  in h), None)
        val_col  = next((h for h in fieldnames if value_col_hint.lower() in h), None)
        if run_col is None or val_col is None:
            print(f"  [warn] expected 'run' and '{value_col_hint}' columns in {path}, got: {fieldnames}")
            return result
        for raw_row in reader:
            row = {k.strip().lower(): v for k, v in raw_row.items()}
            try:
                result[row[run_col].strip()] = float(row[val_col])
            except (ValueError, KeyError):
                continue
    return result


def read_loss_log(path):
    """Return (steps: list[int], losses: list[float])."""
    steps, losses = [], []
    if not os.path.exists(path):
        print(f"  [warn] not found: {path}")
        return steps, losses
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]
        step_col = next((h for h in fieldnames if "step" in h), None)
        loss_col = next((h for h in fieldnames if "loss" in h), None)
        if loss_col is None:
            print(f"  [warn] no 'loss' column in {path}")
            return steps, losses
        for raw_row in reader:
            row = {k.strip().lower(): v for k, v in raw_row.items()}
            try:
                losses.append(float(row[loss_col]))
                steps.append(int(row[step_col]) if step_col else len(steps))
            except (ValueError, KeyError):
                continue
    return steps, losses


def gray_placeholder(h=256, w=256):
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    return img


# ── Figure 1: comparison_table.png ───────────────────────────────────────────

def make_comparison_table(fid_map, clip_map):
    columns = ["Run", "Rank", "LR", "Final Loss", "FID ↓", "Avg CLIP ↑"]
    rows = []
    fid_vals = []

    for run in RUNS:
        _, losses = read_loss_log(run["loss_log"])
        fl = f"{losses[-1]:.4f}" if losses else "N/A"

        fid  = fid_map.get(run["csv_key"])
        clip = clip_map.get(run["csv_key"])
        fid_vals.append(float(fid) if fid is not None else float("inf"))

        rows.append([
            run["name"],
            str(run["rank"]),
            run["lr"],
            fl,
            f"{float(fid):.4f}"  if fid  is not None else "N/A",
            f"{float(clip):.4f}" if clip is not None else "N/A",
        ])

    winner_idx = int(np.argmin(fid_vals))

    fig, ax = plt.subplots(figsize=(10, 2), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for col in range(len(columns)):
        table[(0, col)].set_facecolor("#2c3e50")
        table[(0, col)].set_text_props(color="white", fontweight="bold")

    for row_idx, _ in enumerate(rows):
        color = "#d4efdf" if row_idx == winner_idx else "#fdfefe"
        for col in range(len(columns)):
            table[(row_idx + 1, col)].set_facecolor(color)

    ax.text(0.98, 0.12, "★ Winner (lower FID)",
            transform=ax.transAxes, fontsize=9,
            color="#27ae60", ha="right", va="bottom", fontstyle="italic")

    plt.title("LoRA Run Comparison", fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, "comparison_table.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 2: loss_curves.png ─────────────────────────────────────────────────

def make_loss_curves():
    styles = [
        ("#2980b9", "--"),
        ("#e74c3c", "-"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    for run, (color, ls) in zip(RUNS, styles):
        label = f"Rank {run['rank']}, LR={run['lr']}"
        steps, losses = read_loss_log(run["loss_log"])
        if steps and losses:
            ax.plot(steps, losses, color=color, linestyle=ls,
                    linewidth=2, label=label)
        else:
            ax.plot([], [], color=color, linestyle=ls,
                    linewidth=2, label=f"{label} — log not found")


    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("LoRA Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(EVAL_DIR, "loss_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 3: fid_clip_bar.png ────────────────────────────────────────────────

def make_fid_clip_bar(fid_map, clip_map):
    labels    = [f"Rank {r['rank']}" for r in RUNS]
    fid_vals  = [float(fid_map.get(r["csv_key"], 0))  for r in RUNS]
    clip_vals = [float(clip_map.get(r["csv_key"], 0)) for r in RUNS]
    colors    = ["#2980b9", "#e74c3c"]
    x = np.arange(len(RUNS))

    fig, (ax_fid, ax_clip) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ── FID
    bars = ax_fid.bar(x, fid_vals, width=0.45, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)
    ax_fid.set_xticks(x)
    ax_fid.set_xticklabels(labels, fontsize=12)
    ax_fid.set_ylabel("FID Score", fontsize=12)
    ax_fid.set_title("FID Score per Run", fontsize=13, fontweight="bold")
    ax_fid.grid(axis="y", alpha=0.35, linestyle=":", zorder=0)
    ax_fid.spines["top"].set_visible(False)
    ax_fid.spines["right"].set_visible(False)

    for bar, val in zip(bars, fid_vals):
        ax_fid.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(fid_vals) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    wi = int(np.argmin(fid_vals))
    ax_fid.annotate(
        "lower is better",
        xy=(x[wi], fid_vals[wi] * 0.97),
        xytext=(x[wi], fid_vals[wi] * 0.87),
        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.8),
        color="#27ae60", fontsize=10, ha="center",
    )

    # ── CLIP
    bars2 = ax_clip.bar(x, clip_vals, width=0.45, color=colors,
                        edgecolor="white", linewidth=0.8, zorder=3)
    ax_clip.set_xticks(x)
    ax_clip.set_xticklabels(labels, fontsize=12)
    ax_clip.set_ylabel("Avg CLIP Score", fontsize=12)
    ax_clip.set_title("Avg CLIP Score per Run", fontsize=13, fontweight="bold")
    ax_clip.grid(axis="y", alpha=0.35, linestyle=":", zorder=0)
    ax_clip.spines["top"].set_visible(False)
    ax_clip.spines["right"].set_visible(False)
    y_span = max(clip_vals) - min(clip_vals) if max(clip_vals) != min(clip_vals) else 0.01
    ax_clip.set_ylim(min(clip_vals) - y_span * 2, max(clip_vals) + y_span * 5)
    y_pad = y_span * 0.3

    for bar, val in zip(bars2, clip_vals):
        ax_clip.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_pad,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    wi2 = int(np.argmax(clip_vals))
    ax_clip.annotate(
        "higher is better",
        xy=(x[wi2], clip_vals[wi2] + y_pad * 2),
        xytext=(x[wi2], clip_vals[wi2] + y_pad * 6),
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.8),
        color="#e67e22", fontsize=10, ha="center",
    )

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(colors, labels)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.suptitle("Evaluation Metrics Comparison", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, "fid_clip_bar.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 4: sample_comparison.png ──────────────────────────────────────────

def make_sample_comparison():
    row_labels = [f"Rank {r['rank']}" for r in RUNS]
    n_rows, n_cols = len(RUNS), len(SAMPLE_STEPS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), dpi=150)

    for r_idx, run in enumerate(RUNS):
        sample_dir = os.path.join(OUTPUT_DIR, run["name"], "samples")
        for c_idx, step in enumerate(SAMPLE_STEPS):
            ax = axes[r_idx][c_idx]
            img_path = os.path.join(sample_dir, f"step_{step}.png")

            if os.path.exists(img_path):
                try:
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                except Exception as e:
                    print(f"  [warn] could not read {img_path}: {e}")
                    ax.imshow(gray_placeholder())
                    ax.text(0.5, 0.5, "read error", ha="center", va="center",
                            fontsize=8, color="#7f8c8d", transform=ax.transAxes)
            else:
                ax.imshow(gray_placeholder())
                ax.text(0.5, 0.5, f"step_{step}.png\nnot found",
                        ha="center", va="center", fontsize=9,
                        color="#7f8c8d", transform=ax.transAxes)

            ax.axis("off")

            if r_idx == 0:
                ax.set_title(f"Step {step}", fontsize=12, fontweight="bold", pad=6)

        # row label via ylabel on first column
        axes[r_idx][0].set_ylabel(
            row_labels[r_idx], fontsize=13, fontweight="bold",
            rotation=90, labelpad=10,
        )
        axes[r_idx][0].yaxis.set_label_position("left")
        axes[r_idx][0].yaxis.label.set_visible(True)

    plt.suptitle("Sample Output Comparison by Training Step",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(EVAL_DIR, "sample_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("Reading metric CSVs...")
    fid_map  = read_metric_csv(FID_CSV,  "fid")
    clip_map = read_metric_csv(CLIP_CSV, "clip")

    print("→ Figure 1: comparison table")
    make_comparison_table(fid_map, clip_map)

    print("→ Figure 2: loss curves")
    make_loss_curves()

    print("→ Figure 3: FID / CLIP bar charts")
    make_fid_clip_bar(fid_map, clip_map)


    print("→ Figure 4: sample comparison grid")
    make_sample_comparison()

    print("All figures saved")

