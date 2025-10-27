import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
mm = 54.8                   # inward offset in mm for expected points
n = mm + 15                 # inward offset in mm for inner/start points
x = [4, 2, 5, 3, 3, 3, 4]   # number of equidistant points per edge (list)
y = mm + 30                 # outward perpendicular offset (for outer points)
z_offsets = [1, 30, 30, 30, 30, 30, 30]  # one z-offset PER CORNER (not actually z, it is x,y offset)
circle_radius = 54.8
tank_centre = [-361.65, 737.24]            # x , y centre of tank, was -308.69, 738


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

# --- printed edges ---
""" 
print("edges = [")
for p1, p2 in edges:
    print(f"    [({p1[0]:.2f}, {p1[1]:.2f}), ({p2[0]:.2f}, {p2[1]:.2f})],")
print("]")
"""


# ---------- Geometry helpers ----------
def unit_vector(v): 
    v = np.asarray(v, float); n = np.linalg.norm(v); return v/n if n>0 else v

def normal_vector(v): return np.array([-v[1], v[0]], float)

def offset_edge(p1, p2, dist):
    p1,p2 = np.asarray(p1,float), np.asarray(p2,float)
    v = p2-p1; n = unit_vector(normal_vector(v))*dist
    return [tuple(p1+n), tuple(p2+n)]

def generate_inner_edges(edges, off): return [offset_edge(p1,p2,-off) for p1,p2 in edges]

def equidistant_points_including_endpoints(p1,p2,num):
    p1,p2 = np.asarray(p1,float), np.asarray(p2,float)
    v = p2-p1; L = np.linalg.norm(v)
    if L==0 or num<=0: return []
    if num==1: return [tuple(p1)]
    return [tuple(p1+(i/(num-1))*v) for i in range(num)]

def offset_endpoints_along_edge(p1,p2,z1,z2):
    """ Offset from p1 by z1 and from p2 by z2 along edge """
    p1,p2 = np.asarray(p1,float), np.asarray(p2,float)
    v = p2-p1; L = np.linalg.norm(v)
    if L==0: return []
    d = unit_vector(v)
    return [tuple(p1+min(z1,L)*d), tuple(p2-min(z2,L)*d)]

def offset_points_outward(points,p1,p2,dist):
    v = np.asarray(p2,float)-np.asarray(p1,float); n=unit_vector(normal_vector(v))
    return [tuple(np.asarray(pt,float)+dist*n) for pt in points]

def flip_points(points):
    mirror_offset = (2 * tank_centre[0])  # double the X value (e.g., -494 → +988)
    return [(-px + mirror_offset, py) for (px, py) in points]

# ---------- Validate ----------
if len(x)!=len(edges) or len(z_offsets)!=len(edges):
    raise ValueError("x and z_offsets must match number of edges")

# ---------- Build offset edge sets ----------
inner_edges, expected_edges = generate_inner_edges(edges,n), generate_inner_edges(edges,mm)

# ---------- Generate ordered points ----------
inner_all, expected_all, outer_all = [], [], []
circle_points = []

for i,((inner_p1,inner_p2),(exp_p1,exp_p2)) in enumerate(zip(inner_edges,expected_edges)):
    num = x[i]
    inner_eq = equidistant_points_including_endpoints(inner_p1,inner_p2,num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1,exp_p2,num)

    # z offsets: current corner (i) at start, next corner (i+1) at end
    z1, z2 = z_offsets[i], z_offsets[(i+1)%len(edges)]
    inner_off = offset_endpoints_along_edge(inner_p1,inner_p2,z1,z2)
    exp_off   = offset_endpoints_along_edge(exp_p1,exp_p2,z1,z2)

    # Build ordered list: start offset → middle points → end offset
    if inner_off: inner_all.append(inner_off[0])
    if exp_off:   expected_all.append(exp_off[0])

    if num>2:
        inner_all.extend(inner_eq[1:-1])
        expected_all.extend(exp_eq[1:-1])
        circle_points.extend(exp_eq[1:-1])  # circles on middles

    if inner_off: inner_all.append(inner_off[1])
    if exp_off: 
        expected_all.append(exp_off[1])
        circle_points.extend(exp_off)       # circles on z-offsets

    # Outers in same order
    outer_all.extend(offset_points_outward(inner_all[-(num if num>2 else 2):], inner_p1, inner_p2, y))

# Mirrored sets
inner_all_m, expected_all_m, outer_all_m = flip_points(inner_all), flip_points(expected_all), flip_points(outer_all)
circle_centers, circle_centers_m = circle_points, flip_points(circle_points)

# ---------- Save CSVs ----------
def save_csv(fname, pts):
    with open(fname,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['x','y']); w.writerows(pts)

save_csv('aconcentric_inner.csv', inner_all)
save_csv('aconcentric_expected.csv', expected_all)
save_csv('aconcentric_outer.csv', outer_all)
save_csv('bconcentric_inner.csv', inner_all_m)
save_csv('bconcentric_expected.csv', expected_all_m)
save_csv('bconcentric_outer.csv', outer_all_m)

# ---------- Plot ----------
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(24,12))
for ax in (ax1,ax2): ax.set_aspect('equal')

def plot_edges(ax,edges,color,label):
    for j,(p1,p2) in enumerate(edges):
        ax.plot(*zip(p1,p2),color=color,lw=2,label=label if j==0 else "")

def plot_points(ax,pts,c,label,marker='o',size=20):
    if pts: ax.scatter(*zip(*pts),c=c,s=size,label=label,marker=marker)

def plot_circles(ax,centers,r,color):
    for c in centers: ax.add_patch(plt.Circle(c,r,color=color,fill=False,alpha=0.2))

# A plot
plot_edges(ax1,edges,'gray','Original')
plot_edges(ax1,inner_edges,'blue',f'Inner Offset ({n}mm)')
plot_edges(ax1,expected_edges,'green',f'Expected Offset ({mm}mm)')
plot_points(ax1,inner_all,'blue',"A Inner pts")
plot_points(ax1,expected_all,'green',"A Expected pts")
plot_points(ax1,outer_all,'red',"A Outer pts")
plot_circles(ax1,circle_centers,circle_radius,'green')
ax1.legend(); ax1.grid(True)
ax1.set_title("A Data: Ordered per edge (start offset → middles → end offset)")

# B plot (mirrored)
edges_m=[tuple(flip_points(list(e))) for e in edges]
inner_edges_m=[tuple(flip_points(list(e))) for e in inner_edges]
expected_edges_m=[tuple(flip_points(list(e))) for e in expected_edges]
plot_edges(ax2,edges_m,'gray','Original (mirrored)')
plot_edges(ax2,inner_edges_m,'blue',f'Inner Offset ({n}mm)')
plot_edges(ax2,expected_edges_m,'green',f'Expected Offset ({mm}mm)')
plot_points(ax2,inner_all_m,'blue',"B Inner pts")
plot_points(ax2,expected_all_m,'green',"B Expected pts")
plot_points(ax2,outer_all_m,'red',"B Outer pts")
plot_circles(ax2,circle_centers_m,circle_radius,'green')
ax2.legend(); ax2.grid(True)
ax2.set_title("B Data (mirrored): Ordered per edge (start offset → middles → end offset)")

plt.tight_layout(); plt.show()


# ---------- Mirrored Expected Points with z = -70 ----------
expected_xyz_m = []
for i, (exp_p1, exp_p2) in enumerate(expected_edges):
    num = x[i]
    exp_eq = equidistant_points_including_endpoints(exp_p1, exp_p2, num)

    # Mirror each midpoint
    if num > 2:
        mids = exp_eq[1:-1]
        mids_m = flip_points(mids)
        expected_xyz_m.extend([(pt[0], pt[1], -70) for pt in mids_m])

    # Mirror z-offset endpoints
    z1, z2 = z_offsets[i], z_offsets[(i+1) % len(edges)]
    exp_off = offset_endpoints_along_edge(exp_p1, exp_p2, z1, z2)
    exp_off_m = flip_points(exp_off)
    expected_xyz_m.extend([(pt[0], pt[1], -70) for pt in exp_off_m])

# Save mirrored CSV
with open("expected_a_probe_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z"])
    w.writerows(expected_xyz_m)

# Print nicely
print("\n--- Mirrored Expected Points ---")
for row in expected_xyz_m:
    print(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.2f}")


    # ---------- Export expected points with z (all z = -70) ----------
expected_xyz = []

for i, (exp_p1, exp_p2) in enumerate(expected_edges):
    num = x[i]
    exp_eq = equidistant_points_including_endpoints(exp_p1, exp_p2, num)

    # Only include midpoints (excluding endpoints) + z-offset endpoints
    if num > 2:
        expected_xyz.extend([(pt[0], pt[1], -70) for pt in exp_eq[1:-1]])

    # start and end z-offset points
    z1, z2 = z_offsets[i], z_offsets[(i+1) % len(edges)]
    start_off, end_off = offset_endpoints_along_edge(exp_p1, exp_p2, z1, z2)
    expected_xyz.append((start_off[0], start_off[1], -70))
    expected_xyz.append((end_off[0],   end_off[1],   -70))

# Save CSV
with open("expected_b_probe_points.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x","y","z"])
    w.writerows(expected_xyz)

# Print nicely
for row in expected_xyz:
    print(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.2f}")