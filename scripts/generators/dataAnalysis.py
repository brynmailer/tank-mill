import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Add src/ to Python path so we can import utils
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

import utils  # <-- now Python can find tank-mill/src/utils.py

# ----------------------------
# CONFIG
# ----------------------------
DATA_FOLDER = os.path.join(PROJECT_ROOT, "Data", "tankArchive")
INDEX_COL = None       # set to column name if you need to use a specific column as index
# ----------------------------
# LOAD DATA (via utils)
# ----------------------------
all_data = {}
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        tank_name = os.path.splitext(file)[0]
        df = utils.load_points(os.path.join(DATA_FOLDER, file))
        all_data[tank_name] = df

# ----------------------------
# NUMERICAL COMPARISON
# ----------------------------
summary_stats = {}
for tank, df in all_data.items():
    summary_stats[tank] = df.describe()

summary_df = pd.concat(summary_stats, axis=1)

# ----------------------------
# VISUAL COMPARISON (X-Y point clouds with connecting lines)
# ----------------------------
plt.figure(figsize=(10, 10))

for tank, df in all_data.items():
    plt.plot(
        df["x"], df["y"], 
        marker="o", linestyle="-", linewidth=1.5, markersize=4, 
        label=tank
    )

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Tank X-Y Profiles (utils.load_points)")
plt.legend()
plt.axis("equal")  # keep aspect ratio square
plt.grid(True)
plt.tight_layout()
plt.show()


# ----------------------------
# SECOND PLOT: Only points with y > 1100
# ----------------------------
plt.figure(figsize=(12, 6))  # keep it wider for legend

for tank, df in all_data.items():
    df_filtered = df[df["y"] > 1100]
    if not df_filtered.empty:
        plt.plot(
            df_filtered["x"], df_filtered["y"],
            marker="o", linestyle="-", linewidth=1.5, markersize=4,
            label=tank
        )

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Tank X-Y Profiles (1100 ≤ y ≤ 1140)")
plt.axis("equal")
plt.ylim(1100, 1140)   # <-- restrict y-axis range
plt.grid(True)

# Legend outside plot on the right
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()


# ----------------------------
# PLOT ONLY 'A' TANKS
# ----------------------------
plt.figure(figsize=(12, 6))

for tank, df in all_data.items():
    if "A" in tank:   # only include tanks with "A" in their name
        df_filtered = df[df["y"] > 1100]
        if not df_filtered.empty:
            plt.plot(
                df_filtered["x"], df_filtered["y"],
                marker="o", linestyle="-", linewidth=1.5, markersize=4,
                label=tank
            )

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Tank X-Y Profiles (A tanks, 1100 ≤ y ≤ 1140)")
plt.axis("equal")
plt.ylim(1100, 1140)
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


# ----------------------------
# PLOT ONLY 'B' TANKS
# ----------------------------
plt.figure(figsize=(12, 6))

for tank, df in all_data.items():
    if "B" in tank:   # only include tanks with "B" in their name
        df_filtered = df[df["y"] > 1100]
        if not df_filtered.empty:
            plt.plot(
                df_filtered["x"], df_filtered["y"],
                marker="o", linestyle="-", linewidth=1.5, markersize=4,
                label=tank
            )

plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Tank X-Y Profiles (B tanks, 1100 ≤ y ≤ 1140)")
plt.axis("equal")
plt.ylim(1100, 1140)
plt.grid(True)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()





# ----------------------------
# Function to plot a matched pair (A vs B) in a given axis
# ----------------------------
def plot_pair(ax, tankA, tankB, all_data):
    if tankA in all_data and tankB in all_data:
        dfA = all_data[tankA]
        dfB = all_data[tankB]
        dfA_filt = dfA[dfA["y"] > 1100]
        dfB_filt = dfB[dfB["y"] > 1100]

        ax.plot(dfA_filt["x"], dfA_filt["y"], "b-", label=tankA)
        ax.plot(dfB_filt["x"], dfB_filt["y"], "r-", label=tankB)
        ax.set_ylim(1100, 1140)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{tankA} vs {tankB}")
        ax.grid(True)
        ax.legend(fontsize=8)


# ----------------------------
# Select pairs (update with your real pairs)
# ----------------------------
pairs = [
    ("0211A-20250909_045028", "0211B-20250909_050830"),
    ("0198A-20250909_062428", "0196B-20250911_054817"),
    ("0196A-20250911_060614", "0196B-20250911_054817"),
    ("0191A-20250911_095402", "0191B-20250911_093225"),
    ("0188A-20250911_091228", "0188B-20250911_084938"),
]

# Create figure with 5 vertical subplots
fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True, sharey=True)

for ax, (tankA, tankB) in zip(axes, pairs):
    plot_pair(ax, tankA, tankB, all_data)

fig.suptitle("Zoomed-in Tank Comparisons (A vs B)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for suptitle
plt.show()







# ----------------------------
# AVERAGE MAX/MIN HEIGHTS FOR A AND B TANKS
# ----------------------------
a_max_vals, a_min_vals = [], []
b_max_vals, b_min_vals = [], []

for tank, df in all_data.items():
    if "A" in tank:
        a_max_vals.append(df["y"].max())
        a_min_vals.append(df["y"].min())
    elif "B" in tank:
        b_max_vals.append(df["y"].max())
        b_min_vals.append(df["y"].min())

# Calculate averages safely (avoid errors if empty)
avg_a_max = sum(a_max_vals) / len(a_max_vals) if a_max_vals else float("nan")
avg_a_min = sum(a_min_vals) / len(a_min_vals) if a_min_vals else float("nan")
avg_b_max = sum(b_max_vals) / len(b_max_vals) if b_max_vals else float("nan")
avg_b_min = sum(b_min_vals) / len(b_min_vals) if b_min_vals else float("nan")

print("Average max height (A tanks):", avg_a_max)
print("Average min height (A tanks):", avg_a_min)
print("Average max height (B tanks):", avg_b_max)
print("Average min height (B tanks):", avg_b_min)








# --- First geometry (your original JSON) ---
geom1 = {
    "outcut": [
        [[300.9254, -460.267], [262.4254, -498.767], -38.5, 0.0],
        [[262.4254, -498.767], [-262.4254, -498.767], None, None],
        [[-262.4254, -498.767], [-300.9254, -460.267], 0.0, 38.5],
        [[-300.9254, -460.267], [-300.9254, 315.2341], None, None],
        [[-300.9254, 315.2341], [-295.3343, 328.4859], 18.5, 0.0],
        [[-295.3343, 328.4859], [-125.9188, 493.5188], None, None],
        [[-125.9188, 493.5188], [-113.0099, 498.767], 12.9089, -13.2518],
        [[-113.0099, 498.767], [113.0099, 498.767], None, None],
        [[113.0099, 498.767], [125.9188, 493.5188], 0.0, -18.5],
        [[125.9188, 493.5188], [295.3343, 328.4859], None, None],
        [[295.3343, 328.4859], [300.9254, 315.2341], -12.9089, -13.2518],
        [[300.9254, 315.2341], [300.9254, -460.267], None, None]
    ],
    "groove": [
        [[82.9245, 463.367], [-82.9245, 463.367], None, None],
        [[-82.9245, 463.367], [-122.7678, 447.1684], 0, -57.1],
        [[-122.7678, 447.1684], [-248.2686, 324.9141], None, None],
        [[-248.2686, 324.9141], [-265.5254, 284.0128], 39.8433, -40.9014],
        [[-265.5254, 284.0128], [-265.5254, -181.3969], None, None],
        [[-265.5254, -181.3969], [-191.0202, -309.336], 147.1, 0],
        [[-191.0202, -309.336], [67.3318, -455.9293], None, None],
        [[67.3318, -455.9293], [95.511, -463.367], 28.1792, 49.6623],
        [[95.511, -463.367], [200.4254, -463.367], None, None],
        [[200.4254, -463.367], [265.5254, -398.267], 0, 65.1],
        [[265.5254, -398.267], [265.5254, 284.0128], None, None],
        [[265.5254, 284.0128], [248.2686, 324.9141], -57.1, 0],
        [[248.2686, 324.9141], [122.7678, 447.1684], None, None],
        [[122.7678, 447.1684], [82.9245, 463.367], -39.8433, -40.9014]
    ],
    "holes": [
        [68.2026, -469.0015],
        [159.6754, -473.617],
        [242.903, -460.1448],
        [275.0254, -391.0001],
        [275.0254, -281.0001],
        [275.0254, -168.5001],
        [275.0254, -56.0001],
        [275.0254, 56.4999],
        [275.0254, 168.9999],
        [275.0254, 281.4999],
        [222.0695, 364.0144],
        [141.4845, 442.5147],
        [54.25, 473.617],
        [-55.75, 473.617],
        [-142.9845, 442.5147],
        [-223.5695, 364.0144],
        [-276.5254, 281.4999],
        [-276.5254, 168.9999],
        [-276.5254, 56.4999],
        [-276.5254, -56.0001],
        [-276.5254, -168.5001],
        [-259.9683, -251.6983],
        [-203.1362, -314.6718],
        [-111.9234, -366.4275],
        [-20.7106, -418.1832]
    ]
}

# --- Second geometry (your "over the top" JSON) ---
geom2 = {
    "outcut": [
        [[-300.9254, -460.267], [-262.4254, -498.767], 38.5, 0.0],
        [[-262.4254, -498.767], [262.4254, -498.767], None, None],
        [[262.4254, -498.767], [300.9254, -460.267], 0.0, 38.5],
        [[300.9254, -460.267], [300.9254, 315.2341], None, None],
        [[300.9254, 315.2341], [295.3343, 328.4859], -18.5, 0.0],
        [[295.3343, 328.4859], [125.9188, 493.5188], None, None],
        [[125.9188, 493.5188], [113.0099, 498.767], -12.9089, -13.2518],
        [[113.0099, 498.767], [-113.0099, 498.767], None, None],
        [[-113.0099, 498.767], [-125.9188, 493.5188], 0.0, -18.5],
        [[-125.9188, 493.5188], [-295.3343, 328.4859], None, None],
        [[-295.3343, 328.4859], [-300.9254, 315.2341], 12.9089, -13.2518],
        [[-300.9254, 315.2341], [-300.9254, -460.267], None, None]
    ],
    "groove": [
        [[-82.9245, 463.367], [82.9245, 463.367], None, None],
        [[82.9245, 463.367], [122.7678, 447.1684], 0, -57.1],
        [[122.7678, 447.1684], [248.2686, 324.9141], None, None],
        [[248.2686, 324.9141], [265.5254, 284.0128], -39.8433, -40.9014],
        [[265.5254, 284.0128], [265.5254, -181.3969], None, None],
        [[265.5254, -181.3969], [191.0202, -309.336], -147.1, 0],
        [[191.0202, -309.336], [-67.3318, -455.9293], None, None],
        [[-67.3318, -455.9293], [-95.511, -463.367], -28.1792, 49.6623],
        [[-95.511, -463.367], [-200.4254, -463.367], None, None],
        [[-200.4254, -463.367], [-265.5254, -398.267], 0, 65.1],
        [[-265.5254, -398.267], [-265.5254, 284.0128], None, None],
        [[-265.5254, 284.0128], [-248.2686, 324.9141], 57.1, 0],
        [[-248.2686, 324.9141], [-122.7678, 447.1684], None, None],
        [[-122.7678, 447.1684], [-82.9245, 463.367], 39.8433, -40.9014]
    ],
    "holes": [
        [-68.2026, -469.0015],
        [-159.6754, -473.617],
        [-242.903, -460.1448],
        [-275.0254, -391.0001],
        [-275.0254, -281.0001],
        [-275.0254, -168.5001],
        [-275.0254, -56.0001],
        [-275.0254, 56.4999],
        [-275.0254, 168.9999],
        [-275.0254, 281.4999],
        [-222.0695, 364.0144],
        [-141.4845, 442.5147],
        [-54.25, 473.617],
        [55.75, 473.617],
        [142.9845, 442.5147],
        [223.5695, 364.0144],
        [276.5254, 281.4999],
        [276.5254, 168.9999],
        [276.5254, 56.4999],
        [276.5254, -56.0001],
        [276.5254, -168.5001],
        [259.9683, -251.6983],
        [203.1362, -314.6718],
        [111.9234, -366.4275],
        [20.7106, -418.1832]
    ]
}

# --- Plot both geometries ---
plt.figure(figsize=(10, 10))

# Plot first geometry (solid lines, filled circles)
for seg in geom1["outcut"]:
    plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "b-", lw=2, label="Outcut 1" if "Outcut 1" not in plt.gca().get_legend_handles_labels()[1] else "")
for seg in geom1["groove"]:
    plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "g-", lw=2, label="Groove 1" if "Groove 1" not in plt.gca().get_legend_handles_labels()[1] else "")
plt.scatter([h[0] for h in geom1["holes"]], [h[1] for h in geom1["holes"]], c="r", marker="o", label="Holes 1")

# Plot second geometry (dashed lines, hollow markers)
for seg in geom2["outcut"]:
    plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "b--", lw=2, label="Outcut 2" if "Outcut 2" not in plt.gca().get_legend_handles_labels()[1] else "")
for seg in geom2["groove"]:
    plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], "g--", lw=2, label="Groove 2" if "Groove 2" not in plt.gca().get_legend_handles_labels()[1] else "")
plt.scatter([h[0] for h in geom2["holes"]], [h[1] for h in geom2["holes"]], facecolors="none", edgecolors="r", marker="o", label="Holes 2")

# --- Styling ---
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.title("Comparison of Original vs New Geometry")
plt.legend()
plt.grid(True)
plt.show()
