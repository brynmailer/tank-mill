import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
n = 79    # inward offset in mm for inner points
mm = 64   # inward offset in mm for expected points (different concentric level)
x = [4, 2, 5, 3, 3, 3, 4]   # number of equidistant points per edge (list) -- NOW INCLUDES ENDPOINTS
y = 109   # outward perpendicular offset (for "outer" points, perpendicular to edge)
z = 20     # ALONG-EDGE offset distance from each endpoint to create extra offset endpoint probes

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
    # left-handed normal (CCW) to vector v
    v = np.asarray(v, dtype=float)
    return np.array([-v[1], v[0]], dtype=float)

def offset_edge(p1, p2, distance):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v)) * distance
    return [tuple(p1 + n), tuple(p2 + n)]

def generate_inner_edges(edges, offset_dist):
    # negative offset for "inward" relative to original polygon normal convention
    return [offset_edge(p1, p2, -offset_dist) for p1, p2 in edges]

def equidistant_points_including_endpoints(p1, p2, num_points):
    """
    Return 'num_points' positions along segment p1->p2 INCLUDING endpoints.
    If num_points == 1: returns [p1] (change to midpoint if you prefer).
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    Lvec = p2 - p1
    L = np.linalg.norm(Lvec)
    if L == 0 or num_points <= 0:
        return []
    if num_points == 1:
        return [tuple(p1)]
    # i = 0..N-1; t = i/(N-1) gives endpoints included
    pts = [tuple(p1 + (i / (num_points - 1)) * Lvec) for i in range(num_points)]
    return pts

def offset_endpoints_along_edge(p1, p2, z):
    """
    Create two points offset ALONG the edge direction by z from each endpoint,
    i.e. start+z*dir and end - z*dir (if z <= length; otherwise clamp inside).
    If z <= 0, returns empty list.
    """
    if z <= 0:
        return []
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    L = np.linalg.norm(v)
    if L == 0:
        return []
    d = unit_vector(v) * min(z, L)  # clamp to segment length
    start_off = tuple(p1 + d)
    end_off   = tuple(p2 - d)
    return [start_off, end_off]

def offset_points_outward(points, p1, p2, dist):
    """
    Offset a set of points by 'dist' along the left normal of the edge p1->p2.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v))
    pts = []
    for pt in points:
        pt = np.asarray(pt, dtype=float)
        pts.append(tuple(pt + dist * n))
    return pts

def offset_points_inward_perpendicular(points, p1, p2, dist):
    """
    Offset a set of points by 'dist' perpendicular to the edge p1->p2, inward.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    v = p2 - p1
    n = unit_vector(normal_vector(v))  # left normal
    pts = []
    for pt in points:
        pt = np.asarray(pt, dtype=float)
        # Use negative distance to go inward (toward tank center)
        pts.append(tuple(pt - dist * n))
    return pts

def flip_points(points):
    # mirror across a vertical line x -> -x - 616 (as in your original)
    return [(-px - 616, py) for (px, py) in points]

# ---------- Validate inputs ----------
if len(x) != len(edges):
    raise ValueError(f"Length of points list ({len(x)}) must match number of edges ({len(edges)})")

# ---------- Build offset edge sets ----------
inner_edges    = generate_inner_edges(edges, n)
expected_edges = generate_inner_edges(edges, mm)

# ---------- Generate points ----------
inner_points              = []
inner_points_end_offsets  = []  # start/end offset by z along edge
expected_points           = []
expected_points_end_offsets = []
outer_points              = []
outer_points_end_offsets  = []

for i, ((inner_p1, inner_p2), (exp_p1, exp_p2), base_edge) in enumerate(zip(inner_edges, expected_edges, edges)):
    num = x[i]

    # 1) equidistant points including endpoints, on INNER & EXPECTED edges
    inner_eq = equidistant_points_including_endpoints(inner_p1, inner_p2, num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1, exp_p2, num)

    inner_points.extend(inner_eq)
    expected_points.extend(exp_eq)

    # 2) ALONG-EDGE endpoint offsets (z) for both inner and expected
    inner_end_off = offset_endpoints_along_edge(inner_p1, inner_p2, z)
    exp_end_off   = offset_endpoints_along_edge(exp_p1,   exp_p2,   z)

    inner_points_end_offsets.extend(inner_end_off)
    expected_points_end_offsets.extend(exp_end_off)

    # 3) OUTER points: perpendicular offset of INNER equidistant points and the offset endpoints
    #    (so outer follows the same sampling "pattern" but shifted by +y normal)
    outer_points.extend(offset_points_outward(inner_eq, inner_p1, inner_p2, y))
    outer_points_end_offsets.extend(offset_points_outward(inner_end_off, inner_p1, inner_p2, y))

# Combine for saving/plotting convenience
inner_all    = inner_points + inner_points_end_offsets
expected_all = expected_points + expected_points_end_offsets
outer_all    = outer_points + outer_points_end_offsets

# ---------- Generate points ----------
inner_points              = []
inner_points_end_offsets  = []  # start/end offset by z along edge
expected_points           = []
expected_points_end_offsets = []
outer_points              = []
outer_points_end_offsets  = []

# Store edge information for circle center calculation
edge_info_for_circles = []

for i, ((inner_p1, inner_p2), (exp_p1, exp_p2), base_edge) in enumerate(zip(inner_edges, expected_edges, edges)):
    num = x[i]

    # 1) equidistant points including endpoints, on INNER & EXPECTED edges
    inner_eq = equidistant_points_including_endpoints(inner_p1, inner_p2, num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1, exp_p2, num)

    inner_points.extend(inner_eq)
    expected_points.extend(exp_eq)

    # Store edge info for each expected point
    for pt in exp_eq:
        edge_info_for_circles.append((pt, exp_p1, exp_p2))

    # 2) ALONG-EDGE endpoint offsets (z) for both inner and expected
    inner_end_off = offset_endpoints_along_edge(inner_p1, inner_p2, z)
    exp_end_off   = offset_endpoints_along_edge(exp_p1,   exp_p2,   z)

    inner_points_end_offsets.extend(inner_end_off)
    expected_points_end_offsets.extend(exp_end_off)

    # Store edge info for endpoint offset points too
    for pt in exp_end_off:
        edge_info_for_circles.append((pt, exp_p1, exp_p2))

    # 3) OUTER points: perpendicular offset of INNER equidistant points and the offset endpoints
    #    (so outer follows the same sampling "pattern" but shifted by +y normal)
    outer_points.extend(offset_points_outward(inner_eq, inner_p1, inner_p2, y))
    outer_points_end_offsets.extend(offset_points_outward(inner_end_off, inner_p1, inner_p2, y))

# Combine for saving/plotting convenience
inner_all    = inner_points + inner_points_end_offsets
expected_all = expected_points + expected_points_end_offsets
outer_all    = outer_points + outer_points_end_offsets

# Calculate circle centers (64mm perpendicular inward from expected points)
circle_centers = []
for pt, exp_p1, exp_p2 in edge_info_for_circles:
    centers = offset_points_inward_perpendicular([pt], exp_p1, exp_p2, 64)
    circle_centers.extend(centers)

# ---------- Mirrored sets ----------
inner_all_m   = flip_points(inner_all)
expected_all_m= flip_points(expected_all)
outer_all_m   = flip_points(outer_all)

# Calculate mirrored circle centers
circle_centers_m = []
for i, (pt, exp_p1, exp_p2) in enumerate(edge_info_for_circles):
    # Mirror the edge endpoints
    exp_p1_m = (-exp_p1[0] - 616, exp_p1[1])
    exp_p2_m = (-exp_p2[0] - 616, exp_p2[1])
    # Mirror the point
    pt_m = (-pt[0] - 616, pt[1])
    # Calculate center for mirrored geometry
    centers = offset_points_inward_perpendicular([pt_m], exp_p1_m, exp_p2_m, 64)
    circle_centers_m.extend(centers)

# ---------- Save CSVs ----------
with open('aconcentric_inner.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(inner_all)
with open('aconcentric_expected.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(expected_all)
with open('aconcentric_outer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(outer_all)

with open('bconcentric_inner.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(inner_all_m)
with open('bconcentric_expected.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(expected_all_m)
with open('bconcentric_outer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['x', 'y']); w.writerows(outer_all_m)

# ---------- Create side-by-side plots ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# ---------- Plot A Data (Left subplot) ----------
ax1.set_aspect('equal')

# Plot edges for A data
for p1, p2 in edges:
    xs, ys = zip(p1, p2)
    ax1.plot(xs, ys, color='gray', linewidth=2, label='Original' if p1 == edges[0][0] else "")

for p1, p2 in inner_edges:
    xs, ys = zip(p1, p2)
    ax1.plot(xs, ys, color='blue', linewidth=2, label=f'Inner Offset ({n}mm)' if p1 == inner_edges[0][0] else "")

for p1, p2 in expected_edges:
    xs, ys = zip(p1, p2)
    ax1.plot(xs, ys, color='green', linewidth=2, label=f'Expected Offset ({mm}mm)' if p1 == expected_edges[0][0] else "")

# Plot points for A data
if inner_points:
    xs, ys = zip(*inner_points)
    ax1.scatter(xs, ys, color='blue', s=20, label='A Inner eq pts', marker='o', zorder=3)

if expected_points:
    xs, ys = zip(*expected_points)
    ax1.scatter(xs, ys, color='green', s=20, label='A Expected eq pts', marker='o', zorder=3)

if outer_points:
    xs, ys = zip(*outer_points)
    ax1.scatter(xs, ys, color='red', s=20, label='A Outer eq pts', marker='o', zorder=3)

if inner_points_end_offsets:
    xs, ys = zip(*inner_points_end_offsets)
    ax1.scatter(xs, ys, color='blue', s=60, label='A Inner endpoint+z', marker='x', zorder=4)

if expected_points_end_offsets:
    xs, ys = zip(*expected_points_end_offsets)
    ax1.scatter(xs, ys, color='green', s=60, label='A Expected endpoint+z', marker='x', zorder=4)

if outer_points_end_offsets:
    xs, ys = zip(*outer_points_end_offsets)
    ax1.scatter(xs, ys, color='red', s=60, label='A Outer endpoint+z', marker='x', zorder=4)

# Add circles with centers moved 64mm inward
for center_pt in circle_centers:
    circle = plt.Circle(center_pt, 64, color='green', fill=False, alpha=0.2, linewidth=1)
    ax1.add_patch(circle)

ax1.legend()
ax1.set_title(f"A Data: Concentric Shapes and Points\n"
              f"Equidistant points include endpoints; endpoint offset z = {z} mm\n"
              f"64mm circles with centers moved toward polygon center")
ax1.grid(True)

# ---------- Plot B Data (Right subplot) ----------
ax2.set_aspect('equal')

# Mirror the edges for B plot
edges_m = [flip_points([p1, p2]) for p1, p2 in edges]
inner_edges_m = [flip_points([p1, p2]) for p1, p2 in inner_edges]
expected_edges_m = [flip_points([p1, p2]) for p1, p2 in expected_edges]

# Convert back to edge format
edges_m = [(pts[0], pts[1]) for pts in edges_m]
inner_edges_m = [(pts[0], pts[1]) for pts in inner_edges_m]
expected_edges_m = [(pts[0], pts[1]) for pts in expected_edges_m]

# Plot edges for B data
for p1, p2 in edges_m:
    xs, ys = zip(p1, p2)
    ax2.plot(xs, ys, color='gray', linewidth=2, label='Original (mirrored)' if p1 == edges_m[0][0] else "")

for p1, p2 in inner_edges_m:
    xs, ys = zip(p1, p2)
    ax2.plot(xs, ys, color='blue', linewidth=2, label=f'Inner Offset ({n}mm)' if p1 == inner_edges_m[0][0] else "")

for p1, p2 in expected_edges_m:
    xs, ys = zip(p1, p2)
    ax2.plot(xs, ys, color='green', linewidth=2, label=f'Expected Offset ({mm}mm)' if p1 == expected_edges_m[0][0] else "")

# Mirrored points
inner_points_m = flip_points(inner_points)
inner_points_end_offsets_m = flip_points(inner_points_end_offsets)
expected_points_m = flip_points(expected_points)
expected_points_end_offsets_m = flip_points(expected_points_end_offsets)
outer_points_m = flip_points(outer_points)
outer_points_end_offsets_m = flip_points(outer_points_end_offsets)

# Plot points for B data
if inner_points_m:
    xs, ys = zip(*inner_points_m)
    ax2.scatter(xs, ys, color='blue', s=20, label='B Inner eq pts', marker='o', zorder=3)

if expected_points_m:
    xs, ys = zip(*expected_points_m)
    ax2.scatter(xs, ys, color='green', s=20, label='B Expected eq pts', marker='o', zorder=3)

if outer_points_m:
    xs, ys = zip(*outer_points_m)
    ax2.scatter(xs, ys, color='red', s=20, label='B Outer eq pts', marker='o', zorder=3)

if inner_points_end_offsets_m:
    xs, ys = zip(*inner_points_end_offsets_m)
    ax2.scatter(xs, ys, color='blue', s=60, label='B Inner endpoint+z', marker='x', zorder=4)

if expected_points_end_offsets_m:
    xs, ys = zip(*expected_points_end_offsets_m)
    ax2.scatter(xs, ys, color='green', s=60, label='B Expected endpoint+z', marker='x', zorder=4)

if outer_points_end_offsets_m:
    xs, ys = zip(*outer_points_end_offsets_m)
    ax2.scatter(xs, ys, color='red', s=60, label='B Outer endpoint+z', marker='x', zorder=4)

# Add circles with centers moved 64mm inward (mirrored)
for center_pt in circle_centers_m:
    circle = plt.Circle(center_pt, 64, color='green', fill=False, alpha=0.2, linewidth=1)
    ax2.add_patch(circle)

ax2.legend()
ax2.set_title(f"B Data: Concentric Shapes and Points (Mirrored)\n"
              f"Equidistant points include endpoints; endpoint offset z = {z} mm\n"
              f"64mm circles with centers moved toward polygon center")
ax2.grid(True)

plt.tight_layout()
plt.show()

# ---------- Summary ----------
print(f"Equidistant points per edge (including endpoints): {x}")
print(f"Inner offset: {n}mm, Expected offset: {mm}mm, Outer perpendicular offset: {y}mm, Endpoint along-edge offset z: {z}mm")
print(f"A sets: Inner={len(inner_all)}, Expected={len(expected_all)}, Outer={len(outer_all)}")
print(f"B sets (mirrored): Inner={len(inner_all_m)}, Expected={len(expected_all_m)}, Outer={len(outer_all_m)}")
print(f"64mm circles with centers moved toward polygon center - circles now touch the expected probe points")