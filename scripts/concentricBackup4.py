import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
n = 79    # inward offset in mm for inner points
mm = 64   # inward offset in mm for expected points (different concentric level)
x = [4, 2, 5, 3, 3, 3, 4]   # number of equidistant points per edge (list) -- NOW INCLUDES ENDPOINTS
y = 109   # outward perpendicular offset (for "outer" points, perpendicular to edge)
z = 20    # ALONG-EDGE offset distance from each endpoint to create extra offset endpoint probes

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
    v = np.asarray(v, dtype=float)
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

def offset_endpoints_along_edge(p1, p2, z):
    if z <= 0: return []
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1; L = np.linalg.norm(v)
    if L == 0: return []
    d = unit_vector(v) * min(z, L)
    return [tuple(p1 + d), tuple(p2 - d)]

def offset_points_outward(points, p1, p2, dist):
    v = np.asarray(p2, float) - np.asarray(p1, float)
    n = unit_vector(normal_vector(v))
    return [tuple(np.asarray(pt, float) + dist * n) for pt in points]

def flip_points(points):
    return [(-px - 616, py) for (px, py) in points]

# ---------- Validate ----------
if len(x) != len(edges):
    raise ValueError("Length of x must match number of edges")

# ---------- Build offset edge sets ----------
inner_edges    = generate_inner_edges(edges, n)
expected_edges = generate_inner_edges(edges, mm)

# ---------- Generate points ----------
inner_points, inner_points_end_offsets = [], []
expected_points, expected_points_end_offsets = [], []
outer_points, outer_points_end_offsets = [], []
circle_points_info = []

for i, ((inner_p1, inner_p2), (exp_p1, exp_p2)) in enumerate(zip(inner_edges, expected_edges)):
    num = x[i]
    inner_eq = equidistant_points_including_endpoints(inner_p1, inner_p2, num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1, exp_p2, num)
    inner_points.extend(inner_eq)
    expected_points.extend(exp_eq)

    # Middle expected points get circles
    if num > 2:
        for pt in exp_eq[1:-1]:
            circle_points_info.append((pt, exp_p1, exp_p2))

    inner_end_off = offset_endpoints_along_edge(inner_p1, inner_p2, z)
    exp_end_off   = offset_endpoints_along_edge(exp_p1, exp_p2, z)
    inner_points_end_offsets.extend(inner_end_off)
    expected_points_end_offsets.extend(exp_end_off)

    # endpoint offsets also get circles
    for pt in exp_end_off:
        circle_points_info.append((pt, exp_p1, exp_p2))

    outer_points.extend(offset_points_outward(inner_eq, inner_p1, inner_p2, y))
    outer_points_end_offsets.extend(offset_points_outward(inner_end_off, inner_p1, inner_p2, y))

# Combine
inner_all    = inner_points + inner_points_end_offsets
expected_all = expected_points + expected_points_end_offsets
outer_all    = outer_points + outer_points_end_offsets

# ---------- Circle centers (inward perpendicular, centroid check) ----------
expected_vertices = [p for e in expected_edges for p in e]
centroid = np.mean(np.unique(np.array(expected_vertices), axis=0), axis=0)

def inward_center_for_point(pt, p1, p2, radius, poly_centroid):
    pt = np.asarray(pt, float)
    v  = np.asarray(p2, float) - np.asarray(p1, float)
    n  = unit_vector(normal_vector(v))
    if np.dot(n, (poly_centroid - pt)) < 0: n = -n
    return tuple(pt + radius * n)

circle_centers = [
    inward_center_for_point(pt, exp_p1, exp_p2, 64.0, centroid)
    for (pt, exp_p1, exp_p2) in circle_points_info
]

# ---------- Mirrored sets ----------
inner_all_m    = flip_points(inner_all)
expected_all_m = flip_points(expected_all)
outer_all_m    = flip_points(outer_all)

expected_edges_m = [tuple(flip_points(list(e))) for e in expected_edges]
expected_vertices_m = [p for e in expected_edges_m for p in e]
centroid_m = np.mean(np.unique(np.array(expected_vertices_m), axis=0), axis=0)

circle_centers_m = [
    inward_center_for_point(
        (-pt[0] - 616, pt[1]),
        (-exp_p1[0] - 616, exp_p1[1]),
        (-exp_p2[0] - 616, exp_p2[1]),
        64.0,
        centroid_m
    )
    for (pt, exp_p1, exp_p2) in circle_points_info
]

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
ax1.set_aspect('equal'); ax2.set_aspect('equal')

# A plot
for p1, p2 in edges: ax1.plot(*zip(p1,p2), color='gray', lw=2, label='Original' if p1 == edges[0][0] else "")
for p1, p2 in inner_edges: ax1.plot(*zip(p1,p2), 'b', lw=2, label=f'Inner Offset ({n}mm)' if p1 == inner_edges[0][0] else "")
for p1, p2 in expected_edges: ax1.plot(*zip(p1,p2), 'g', lw=2, label=f'Expected Offset ({mm}mm)' if p1 == expected_edges[0][0] else "")

ax1.scatter(*zip(*inner_points), c='b', s=20, label="A Inner eq pts", marker='o')
ax1.scatter(*zip(*expected_points), c='g', s=20, label="A Expected eq pts", marker='o')
ax1.scatter(*zip(*outer_points), c='r', s=20, label="A Outer eq pts", marker='o')

ax1.scatter(*zip(*inner_points_end_offsets), c='b', s=60, label="A Inner endpoint+z", marker='x')
ax1.scatter(*zip(*expected_points_end_offsets), c='g', s=60, label="A Expected endpoint+z", marker='x')
ax1.scatter(*zip(*outer_points_end_offsets), c='r', s=60, label="A Outer endpoint+z", marker='x')

for c in circle_centers: ax1.add_patch(plt.Circle(c, 64, color='g', fill=False, alpha=0.2))

ax1.legend(); ax1.grid(True)
ax1.set_title("A Data: Concentric Shapes and Points\nCircles inward perpendicular; endpoint offset z included")

# B plot
edges_m = [tuple(flip_points(list(e))) for e in edges]
inner_edges_m = [tuple(flip_points(list(e))) for e in inner_edges]
expected_edges_m = [tuple(flip_points(list(e))) for e in expected_edges]

for p1, p2 in edges_m: ax2.plot(*zip(p1,p2), color='gray', lw=2, label='Original (mirrored)' if p1 == edges_m[0][0] else "")
for p1, p2 in inner_edges_m: ax2.plot(*zip(p1,p2), 'b', lw=2, label=f'Inner Offset ({n}mm)' if p1 == inner_edges_m[0][0] else "")
for p1, p2 in expected_edges_m: ax2.plot(*zip(p1,p2), 'g', lw=2, label=f'Expected Offset ({mm}mm)' if p1 == expected_edges_m[0][0] else "")

ax2.scatter(*zip(*flip_points(inner_points)), c='b', s=20, label="B Inner eq pts", marker='o')
ax2.scatter(*zip(*flip_points(expected_points)), c='g', s=20, label="B Expected eq pts", marker='o')
ax2.scatter(*zip(*flip_points(outer_points)), c='r', s=20, label="B Outer eq pts", marker='o')

ax2.scatter(*zip(*flip_points(inner_points_end_offsets)), c='b', s=60, label="B Inner endpoint+z", marker='x')
ax2.scatter(*zip(*flip_points(expected_points_end_offsets)), c='g', s=60, label="B Expected endpoint+z", marker='x')
ax2.scatter(*zip(*flip_points(outer_points_end_offsets)), c='r', s=60, label="B Outer endpoint+z", marker='x')

for c in circle_centers_m: ax2.add_patch(plt.Circle(c, 64, color='g', fill=False, alpha=0.2))

ax2.legend(); ax2.grid(True)
ax2.set_title("B Data: Concentric Shapes and Points (Mirrored)\nCircles inward perpendicular; endpoint offset z included")

plt.tight_layout(); plt.show()
