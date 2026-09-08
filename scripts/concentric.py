import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
mm = 54.8                   # inward offset in mm for expected points
n = mm + 15                 # inward offset in mm for inner/start points
x = [5, 2, 9, 3, 3, 3, 7]   # number of equidistant points per edge (list)
y = mm + 50                 # outward perpendicular offset (for outer points)
z_offsets = [0, 20, 20, 20, 20, 20, 20]  # one z-offset PER CORNER (not actually z, it is x,y offset)
circle_radius = 54.8
tank_centre = [-361.65, 737.24]          # x , y centre of tank


edges_centered = [
    [(182.58, -294.46), (-75.77, -441.06)],
    [(-95.51, -446.27), (-200.43, -446.27)],
    [(-248.43, -398.27), (-248.43, 284.01)],
    [(-236.34, 312.67), (-110.84, 434.92)],
    [(-82.92, 446.27), (82.92, 446.27)],
    [(110.84, 434.92), (236.34, 312.67)],
    [(248.43, 284.01), (248.43, -181.40)]
]

# Automatically offset all edges relative to tank_centre
edges = [
    [((p1[0] + tank_centre[0]), (p1[1] + tank_centre[1])),
     ((p2[0] + tank_centre[0]), (p2[1] + tank_centre[1]))]
    for p1, p2 in edges_centered
]

# ---------- Geometry helpers ----------
def unit_vector(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 0 else v

def normal_vector(v):
    return np.array([-v[1], v[0]], float)

def offset_edge(p1, p2, dist):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1
    n = unit_vector(normal_vector(v)) * dist
    return [tuple(p1 + n), tuple(p2 + n)]

def generate_inner_edges(edges, off):
    return [offset_edge(p1, p2, -off) for p1, p2 in edges]

def equidistant_points_including_endpoints(p1, p2, num):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L == 0 or num <= 0:
        return []
    if num == 1:
        return [tuple(p1)]
    return [tuple(p1 + (i / (num - 1)) * v) for i in range(num)]

def offset_endpoints_along_edge(p1, p2, z1, z2):
    """Offset from p1 by z1 and from p2 by z2 along edge"""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L == 0:
        return []
    d = unit_vector(v)
    return [tuple(p1 + min(z1, L) * d), tuple(p2 - min(z2, L) * d)]

def offset_points_outward(points, p1, p2, dist):
    v = np.asarray(p2, float) - np.asarray(p1, float)
    n = unit_vector(normal_vector(v))
    return [tuple(np.asarray(pt, float) + dist * n) for pt in points]

def flip_points(points):
    mirror_offset = (2 * tank_centre[0])  # double the X value
    return [(-px + mirror_offset, py) for (px, py) in points]

def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ---------- Validate ----------
if len(x) != len(edges) or len(z_offsets) != len(edges):
    raise ValueError("x and z_offsets must match number of edges")

# ---------- Build offset edge sets ----------
inner_edges  = generate_inner_edges(edges, n)
expected_edges = generate_inner_edges(edges, mm)

# ---------- Generate ordered points ----------
inner_all, expected_all, outer_all = [], [], []
circle_points = []

# Store per-edge expected points for later CSV export
expected_eq_per_edge = []

for i, ((inner_p1, inner_p2), (exp_p1, exp_p2)) in enumerate(zip(inner_edges, expected_edges)):
    num = x[i]

    # --- compute z-offset start/end on both inner and expected edges ---
    z1, z2 = z_offsets[i], z_offsets[(i + 1) % len(edges)]

    inner_start, inner_end = offset_endpoints_along_edge(inner_p1, inner_p2, z1, z2)
    exp_start,   exp_end   = offset_endpoints_along_edge(exp_p1,   exp_p2,   z1, z2)

    # --- generate equidistant points BETWEEN these offset endpoints ---
    inner_eq = equidistant_points_including_endpoints(inner_start, inner_end, num)
    exp_eq   = equidistant_points_including_endpoints(exp_start,   exp_end,   num)

    # store per-edge for later use
    expected_eq_per_edge.append(exp_eq)

    # store all points (for plotting/CSV)
    inner_all.extend(inner_eq)
    expected_all.extend(exp_eq)
    circle_points.extend(exp_eq)  # circles on all expected points

    # --- generate outer points corresponding to inner_eq ---
    outer_eq = offset_points_outward(inner_eq, inner_p1, inner_p2, y)
    outer_all.extend(outer_eq)

    # --- compute and print distances between consecutive expected points ---
    print(f"\n>>> Edge {i+1} distances (expected points, after z-offset):")
    for j in range(len(exp_eq) - 1):
        dseg = dist(exp_eq[j], exp_eq[j + 1])
        print(f"  d{j} = {dseg:.3f} mm")


# Mirrored sets
inner_all_m    = flip_points(inner_all)
expected_all_m = flip_points(expected_all)
outer_all_m    = flip_points(outer_all)
circle_centers   = circle_points
circle_centers_m = flip_points(circle_points)

# ---------- Save CSVs ----------
def save_csv(fname, pts):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y'])
        w.writerows(pts)

save_csv('aconcentric_inner.csv', inner_all)
save_csv('aconcentric_expected.csv', expected_all)
save_csv('aconcentric_outer.csv', outer_all)
save_csv('bconcentric_inner.csv', inner_all_m)
save_csv('bconcentric_expected.csv', expected_all_m)
save_csv('bconcentric_outer.csv', outer_all_m)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
for ax in (ax1, ax2):
    ax.set_aspect('equal')

def plot_edges(ax, edges_, color, label):
    for j, (p1, p2) in enumerate(edges_):
        ax.plot(*zip(p1, p2), color=color, lw=2, label=label if j == 0 else "")

def plot_points(ax, pts, c, label, marker='o', size=20):
    if pts:
        ax.scatter(*zip(*pts), c=c, s=size, label=label, marker=marker)

def plot_circles(ax, centers, r, color):
    for c_ in centers:
        ax.add_patch(plt.Circle(c_, r, color=color, fill=False, alpha=0.2))

# A plot
plot_edges(ax1, edges, 'gray', 'Original')
plot_edges(ax1, inner_edges, 'blue',  f'Inner Offset ({n}mm)')
plot_edges(ax1, expected_edges, 'green', f'Expected Offset ({mm}mm)')
plot_points(ax1, inner_all, 'blue',  "A Inner pts")
plot_points(ax1, expected_all, 'green', "A Expected pts")
plot_points(ax1, outer_all, 'red',  "A Outer pts")
plot_circles(ax1, circle_centers, circle_radius, 'green')
ax1.legend()
ax1.grid(True)
ax1.set_title("A Data: z-offset edges → equidistant points")

# B plot (mirrored)
edges_m          = [tuple(flip_points(list(e))) for e in edges]
inner_edges_m    = [tuple(flip_points(list(e))) for e in inner_edges]
expected_edges_m = [tuple(flip_points(list(e))) for e in expected_edges]

plot_edges(ax2, edges_m, 'gray', 'Original (mirrored)')
plot_edges(ax2, inner_edges_m, 'blue',  f'Inner Offset ({n}mm)')
plot_edges(ax2, expected_edges_m, 'green', f'Expected Offset ({mm}mm)')
plot_points(ax2, inner_all_m, 'blue',  "B Inner pts")
plot_points(ax2, expected_all_m, 'green', "B Expected pts")
plot_points(ax2, outer_all_m, 'red',  "B Outer pts")
plot_circles(ax2, circle_centers_m, circle_radius, 'green')
ax2.legend()
ax2.grid(True)
ax2.set_title("B Data (mirrored): z-offset edges → equidistant points")

plt.tight_layout()
plt.show()

# ---------- Mirrored Expected Points with z = -70 ----------
expected_xyz_m = []

for i in range(len(expected_edges)):
    exp_eq = expected_eq_per_edge[i]
    num = len(exp_eq)

    # midpoints (excluding endpoints) if any
    if num > 2:
        mids = exp_eq[1:-1]
        mids_m = flip_points(mids)
        expected_xyz_m.extend([(pt[0], pt[1], -70) for pt in mids_m])

    # z-offset endpoints (already in exp_eq[0] and exp_eq[-1])
    endpoints = [exp_eq[0], exp_eq[-1]]
    endpoints_m = flip_points(endpoints)
    expected_xyz_m.extend([(pt[0], pt[1], -70) for pt in endpoints_m])

# Save mirrored CSV
with open("expected_a_probe_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "z"])
    w.writerows(expected_xyz_m)

# Print nicely
print("\n--- Mirrored Expected Points (z = -70) ---")
for row in expected_xyz_m:
    print(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.2f}")

# ---------- Export expected points with z (all z = -70) ----------
expected_xyz = []

for i in range(len(expected_edges)):
    exp_eq = expected_eq_per_edge[i]
    num = len(exp_eq)

    # midpoints (excluding endpoints)
    if num > 2:
        expected_xyz.extend([(pt[0], pt[1], -70) for pt in exp_eq[1:-1]])

    # z-offset endpoints (exp_eq[0], exp_eq[-1])
    expected_xyz.append((exp_eq[0][0],   exp_eq[0][1],   -70))
    expected_xyz.append((exp_eq[-1][0],  exp_eq[-1][1],  -70))

# Save CSV
with open("expected_b_probe_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y", "z"])
    w.writerows(expected_xyz)

# Print nicely
print("\n--- Expected Points (z = -70) ---")
for row in expected_xyz:
    print(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.2f}")
