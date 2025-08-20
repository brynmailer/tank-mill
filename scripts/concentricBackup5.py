import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
n = 79    # inward offset in mm for inner points
mm = 64   # inward offset in mm for expected points (different concentric level)
x = [4, 2, 5, 3, 3, 3, 4]   # number of equidistant points per edge (includes endpoints)
y = 109   # outward perpendicular offset (for "outer" points)
z_vertices = [10, 20, 80, 40, 40, 40, 40]  # z offsets per vertex (length = number of vertices)
probe_radius = 64

edges = [
    [(-126.10668204909999, 442.7755669043), (-384.4586113317, 296.1822802327)],
    [(-404.198871072, 290.9719647008), (-509.11323920999996, 290.9719647008)],
    [(-557.11323921, 338.9719647008), (-557.11323921, 1021.2517331501999)],
    [(-545.0244766326, 1049.9041820078), (-419.5236626445, 1172.1584100271)],
    [(-391.6124252219, 1183.5059611695), (-225.763340193, 1183.5059611695)],
    [(-197.8521027704, 1172.1584100271), (-72.3512887823, 1049.9041820078)],
    [(-60.26252620490001, 1021.2517331501999), (-60.26252620490001, 555.8420414254)]
]

# ---------- Geometry helpers ----------
def unit_vector(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def normal_vector(v):
    return np.array([-v[1], v[0]], dtype=float)

def offset_edge(p1, p2, distance):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v)) * distance
    return [tuple(p1 + n), tuple(p2 + n)]

def generate_inner_edges(edges, offset_dist):
    return [offset_edge(p1, p2, -offset_dist) for p1, p2 in edges]

def equidistant_points_including_endpoints(p1, p2, num_points):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    Lvec = p2 - p1
    if np.linalg.norm(Lvec) == 0 or num_points <= 0:
        return []
    if num_points == 1:
        return [tuple(p1)]
    return [tuple(p1 + (i / (num_points - 1)) * Lvec) for i in range(num_points)]

def offset_along_edge_point(p1, p2, z, at_start=True):
    if z <= 0:
        return None
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L == 0:
        return None
    d = unit_vector(v) * min(z, L)
    return tuple(p1 + d) if at_start else tuple(p2 - d)

def offset_points_outward(points, p1, p2, dist):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v))
    return [tuple(np.asarray(pt, dtype=float) + dist * n) for pt in points]

def offset_points_inward_perpendicular(points, p1, p2, dist):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v))
    return [tuple(np.asarray(pt, dtype=float) - dist * n) for pt in points]

def flip_points(points):
    return [(-px - 616, py) for (px, py) in points]

# ---------- Validate ----------
if len(x) != len(edges):
    raise ValueError("Length of x must match number of edges")
if len(z_vertices) != len(edges):
    raise ValueError("Length of z_vertices must match number of vertices")

# ---------- Build offset edges ----------
inner_edges    = generate_inner_edges(edges, n)
expected_edges = generate_inner_edges(edges, mm)

# ---------- Generate points ----------
inner_points, expected_points, outer_points = [], [], []
inner_points_end_offsets, expected_points_end_offsets, outer_points_end_offsets = [], [], []
circle_points_info = []

for i, ((inner_p1, inner_p2), (exp_p1, exp_p2)) in enumerate(zip(inner_edges, expected_edges)):
    num = x[i]

    # Equidistant points
    inner_eq = equidistant_points_including_endpoints(inner_p1, inner_p2, num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1, exp_p2, num)
    inner_points.extend(inner_eq)
    expected_points.extend(exp_eq)

    if num > 2:
        for pt in exp_eq[1:-1]:
            circle_points_info.append((pt, exp_p1, exp_p2))

    # Vertex z offsets
    z_start = z_vertices[i]
    z_end   = z_vertices[(i + 1) % len(z_vertices)]

    inner_start_off = offset_along_edge_point(inner_p1, inner_p2, z_start, at_start=True)
    inner_end_off   = offset_along_edge_point(inner_p1, inner_p2, z_end,   at_start=False)
    exp_start_off   = offset_along_edge_point(exp_p1,   exp_p2,   z_start, at_start=True)
    exp_end_off     = offset_along_edge_point(exp_p1,   exp_p2,   z_end,   at_start=False)

    for pt in [inner_start_off, inner_end_off]:
        if pt: inner_points_end_offsets.append(pt)
    for pt in [exp_start_off, exp_end_off]:
        if pt:
            expected_points_end_offsets.append(pt)
            circle_points_info.append((pt, exp_p1, exp_p2))

    outer_points.extend(offset_points_outward(inner_eq, inner_p1, inner_p2, y))
    outer_points_end_offsets.extend(offset_points_outward(
        [p for p in [inner_start_off, inner_end_off] if p], inner_p1, inner_p2, y))

# Combine
inner_all    = inner_points + inner_points_end_offsets
expected_all = expected_points + expected_points_end_offsets
outer_all    = outer_points + outer_points_end_offsets

# Circle centers
circle_centers = []
for pt, exp_p1, exp_p2 in circle_points_info:
    circle_centers.extend(offset_points_inward_perpendicular([pt], exp_p1, exp_p2, probe_radius))

# ---------- Mirrored ----------
inner_all_m    = flip_points(inner_all)
expected_all_m = flip_points(expected_all)
outer_all_m    = flip_points(outer_all)
circle_centers_m = flip_points(circle_centers)

# ---------- Save CSVs ----------
def save_csv(name, points):
    with open(name, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(points)

save_csv('aconcentric_inner.csv', inner_all)
save_csv('aconcentric_expected.csv', expected_all)
save_csv('aconcentric_outer.csv', outer_all)
save_csv('bconcentric_inner.csv', inner_all_m)
save_csv('bconcentric_expected.csv', expected_all_m)
save_csv('bconcentric_outer.csv', outer_all_m)

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

def plot_edges(ax, edges, color, label):
    for j, (p1, p2) in enumerate(edges):
        xs, ys = zip(p1, p2)
        ax.plot(xs, ys, color=color, linewidth=2, label=label if j == 0 else "")

def plot_points(ax, points, color, label, marker='o', size=20, zorder=3):
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, color=color, s=size, label=label, marker=marker, zorder=zorder)

def plot_circles(ax, centers, radius, color):
    for c in centers:
        ax.add_patch(plt.Circle(c, radius, color=color, fill=False, alpha=0.2, linewidth=1))

# A data
ax1.set_aspect('equal')
plot_edges(ax1, edges, 'gray', 'Original')
plot_edges(ax1, inner_edges, 'blue', f'Inner Offset ({n}mm)')
plot_edges(ax1, expected_edges, 'green', f'Expected Offset ({mm}mm)')
plot_points(ax1, inner_points, 'blue', 'A Inner eq pts')
plot_points(ax1, expected_points, 'green', 'A Expected eq pts')
plot_points(ax1, outer_points, 'red', 'A Outer eq pts')
plot_points(ax1, inner_points_end_offsets, 'blue', 'A Inner endpoint+z', marker='x', size=60, zorder=4)
plot_points(ax1, expected_points_end_offsets, 'green', 'A Expected endpoint+z', marker='x', size=60, zorder=4)
plot_points(ax1, outer_points_end_offsets, 'red', 'A Outer endpoint+z', marker='x', size=60, zorder=4)
plot_circles(ax1, circle_centers, probe_radius, 'green')
ax1.legend(); ax1.grid(True)
ax1.set_title("A Data with vertex-based z offsets")

# B data
ax2.set_aspect('equal')
edges_m    = [(flip_points([p1, p2])[0], flip_points([p1, p2])[1]) for p1, p2 in edges]
inner_m    = [(flip_points([p1, p2])[0], flip_points([p1, p2])[1]) for p1, p2 in inner_edges]
expected_m = [(flip_points([p1, p2])[0], flip_points([p1, p2])[1]) for p1, p2 in expected_edges]

plot_edges(ax2, edges_m, 'gray', 'Original (mirrored)')
plot_edges(ax2, inner_m, 'blue', f'Inner Offset ({n}mm)')
plot_edges(ax2, expected_m, 'green', f'Expected Offset ({mm}mm)')
plot_points(ax2, flip_points(inner_points), 'blue', 'B Inner eq pts')
plot_points(ax2, flip_points(expected_points), 'green', 'B Expected eq pts')
plot_points(ax2, flip_points(outer_points), 'red', 'B Outer eq pts')
plot_points(ax2, flip_points(inner_points_end_offsets), 'blue', 'B Inner endpoint+z', marker='x', size=60, zorder=4)
plot_points(ax2, flip_points(expected_points_end_offsets), 'green', 'B Expected endpoint+z', marker='x', size=60, zorder=4)
plot_points(ax2, flip_points(outer_points_end_offsets), 'red', 'B Outer endpoint+z', marker='x', size=60, zorder=4)
plot_circles(ax2, circle_centers_m,probe_radius, 'green')
ax2.legend(); ax2.grid(True)
ax2.set_title("B Data (mirrored) with vertex-based z offsets")

plt.tight_layout()
plt.show()

# ---------- Summary ----------
print(f"Equidistant points per edge (including endpoints): {x}")
print(f"z offsets per vertex: {z_vertices}")
print(f"A sets: Inner={len(inner_all)}, Expected={len(expected_all)}, Outer={len(outer_all)}")
print(f"B sets: Inner={len(inner_all_m)}, Expected={len(expected_all_m)}, Outer={len(outer_all_m)}")
print(f"Circles created: {len(circle_centers)} per side")
