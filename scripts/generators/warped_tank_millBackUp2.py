import sys
sys.path.append("./src")

import matplotlib.pyplot as plot
import numpy as np
import json
from shapely.geometry import LineString, Point as ShapelyPoint
from scipy.interpolate import splprep, splev, interp1d, Rbf
from matplotlib.collections import LineCollection

import utils

# ---- Settings ----
path = "./data/a_points.csv"
groove_depth = 5     # Groove cut depth (mm)
outcut_depth = 12    # Outcut depth (mm)
cut_step = 3         # Step-down increment (mm)

# ---- 1. Read probe points ----
probed_points = utils.load_points(path)

# ---- 2. Load segment definitions from JSON ----
with open("./data/b_tank_mill_features.json", "r") as f:
    features = json.load(f)

# Convert JSON format to tuple format expected by utils.build_path
def convert_segments(json_segments):
    segments = []
    for segment in json_segments:
        start_point = tuple(segment[0])
        end_point = tuple(segment[1])
        i_param = segment[2]
        j_param = segment[3]
        segments.append((start_point, end_point, i_param, j_param))
    return segments

groove_segments = convert_segments(features["groove"])
outcut_segments = convert_segments(features["outcut"])

# ---- 3. Center paths ----
def center_path(x, y, valid_overlay):
    bbox_center_x = (valid_overlay["x"].min() + valid_overlay["x"].max()) / 2
    bbox_center_y = (valid_overlay["y"].min() + valid_overlay["y"].max()) / 2
    path_center_x = (np.min(x) + np.max(x)) / 2
    path_center_y = (np.min(y) + np.max(y)) / 2
    offset_x = bbox_center_x - path_center_x
    offset_y = bbox_center_y - path_center_y

    print(f"bbox_center: ({bbox_center_x}, {bbox_center_y})")
    print(f"path_center: ({path_center_x}, {path_center_y})")
    print(f"offset: ({offset_x}, {offset_y})")

    return x + offset_x, y + offset_y

print("Building groove path")
groove_x, groove_y = utils.build_path(groove_segments, utils.arc_g3)
groove_x, groove_y = center_path(groove_x, groove_y, probed_points)
orig_groove_x, orig_groove_y = np.copy(groove_x), np.copy(groove_y)

print("\n")

print("Building outcut path")
outcut_x, outcut_y = utils.build_path(outcut_segments, utils.arc_g2)
outcut_x, outcut_y = center_path(outcut_x, outcut_y, probed_points)
orig_outcut_x, orig_outcut_y = np.copy(outcut_x), np.copy(outcut_y)

# ---- 4. Groove Warping ----
baseline = 80.0
gcode_path = LineString(list(zip(groove_x, groove_y)))

def residual(px, py):
    p = ShapelyPoint(px, py)
    d = p.distance(gcode_path.interpolate(gcode_path.project(p)))  # unsigned distance
    return d - baseline  # <-- flip sign: if d>baseline, residual is negative (pull inward)

rbf = Rbf(
    probed_points["x"], probed_points["y"],
    [residual(row["x"], row["y"]) for _, row in probed_points.iterrows()],
    function='linear'
)


warped_x, warped_y = [], []
for i in range(len(groove_x)):
    x, y = groove_x[i], groove_y[i]
    if i == 0:
        dx, dy = groove_x[i+1] - x, groove_y[i+1] - y
    elif i == len(groove_x)-1:
        dx, dy = x - groove_x[i-1], y - groove_y[i-1]
    else:
        dx, dy = groove_x[i+1] - groove_x[i-1], groove_y[i+1] - groove_y[i-1]
    tangent = np.array([dx, dy])
    norm = np.linalg.norm(tangent)
    if norm == 0:
        nx, ny = 0, 1
    else:
        tangent /= norm
        nx, ny = -tangent[1], tangent[0]
    delta = rbf(x, y)
    warped_x.append(x + delta * nx)
    warped_y.append(y + delta * ny)

def clean_path(x, y):
    points = np.column_stack([x, y])
    points = points[~np.isnan(points).any(axis=1)]
    points = points[~np.isinf(points).any(axis=1)]
    diffs = np.diff(points, axis=0)
    mask = np.any(diffs != 0, axis=1)
    cleaned = np.vstack([points[0], points[1:][mask]])
    return cleaned[:,0], cleaned[:,1]

warped_x, warped_y = clean_path(np.array(warped_x), np.array(warped_y))

# Spline smoothing for groove
tck, u = splprep([warped_x, warped_y], s=5.0, per=True)
unew = np.linspace(0, 1.0, 1000)
smooth_groove_x, smooth_groove_y = splev(unew, tck)
gcode_path = LineString(list(zip(smooth_groove_x, smooth_groove_y)))

# Depth interpolation along groove
probe_proj_dist = [gcode_path.project(ShapelyPoint(row["x"], row["y"])) for idx, row in probed_points.iterrows()]
path_proj_dist = [gcode_path.project(ShapelyPoint(x, y)) for x, y in zip(smooth_groove_x, smooth_groove_y)]
sort_idx = np.argsort(probe_proj_dist)
probe_proj_dist_sorted = np.array(probe_proj_dist)[sort_idx]
z_values_sorted = np.array(probed_points["z"])[sort_idx]
z_interp = interp1d(probe_proj_dist_sorted, z_values_sorted, kind='linear', fill_value='extrapolate')
surface_z = z_interp(path_proj_dist)  # Tank surface at each groove path point

# ---- 5. Outcut Warping (use groove delta for same fraction of path) ----
groove_path_obj = LineString(list(zip(smooth_groove_x, smooth_groove_y)))
outcut_path_obj = LineString(list(zip(outcut_x, outcut_y)))
groove_len = groove_path_obj.length
outcut_len = outcut_path_obj.length
n_samples = 1000
groove_distances = np.linspace(0, groove_len, n_samples)
outcut_distances = np.linspace(0, outcut_len, n_samples)
groove_pts = np.array([groove_path_obj.interpolate(d) for d in groove_distances])
outcut_pts = np.array([outcut_path_obj.interpolate(d) for d in outcut_distances])
groove_frac = groove_distances / groove_len
outcut_frac = outcut_distances / outcut_len

# Deformation delta for each groove point
groove_delta = np.array([rbf(pt.x, pt.y) for pt in groove_pts])
delta_interp = interp1d(np.linspace(0, 1, len(groove_delta)), groove_delta, kind='linear', fill_value='extrapolate')
outcut_delta = delta_interp(outcut_frac)

# Offset outcut by delta in normal-to-tank-center direction
tank_center_x = (probed_points["x"].min() + probed_points["x"].max()) / 2
tank_center_y = (probed_points["y"].min() + probed_points["y"].max()) / 2
outcut_warped_x, outcut_warped_y = [], []
for pt, delta in zip(outcut_pts, outcut_delta):
    x0, y0 = pt.x, pt.y
    angle = np.arctan2(y0 - tank_center_y, x0 - tank_center_x)
    r_base = np.hypot(x0 - tank_center_x, y0 - tank_center_y)
    new_r = r_base + delta
    wx = tank_center_x + new_r * np.cos(angle)
    wy = tank_center_y + new_r * np.sin(angle)
    outcut_warped_x.append(wx)
    outcut_warped_y.append(wy)
outcut_warped_x, outcut_warped_y = np.array(outcut_warped_x), np.array(outcut_warped_y)

# Depth along outcut
outcut_path_obj_warped = LineString(list(zip(outcut_warped_x, outcut_warped_y)))
outcut_proj_dist = [outcut_path_obj_warped.project(ShapelyPoint(x, y)) for x, y in zip(outcut_warped_x, outcut_warped_y)]
outcut_z = z_interp(np.clip(outcut_proj_dist, probe_proj_dist_sorted[0], probe_proj_dist_sorted[-1]))
final_outcut_z = outcut_z - outcut_depth

# ---- 6. Drill points, center and warp (use groove delta at nearest groove fraction) ----
drill_points = np.array(features["holes"])

bbox_center_x = (probed_points["x"].min() + probed_points["x"].max()) / 2
bbox_center_y = (probed_points["y"].min() + probed_points["y"].max()) / 2
drill_center_x = (np.min(drill_points[:,0]) + np.max(drill_points[:,0])) / 2
drill_center_y = (np.min(drill_points[:,1]) + np.max(drill_points[:,1])) / 2
drill_offset_x = bbox_center_x - drill_center_x
drill_offset_y = bbox_center_y - drill_center_y
centered_drill_points = drill_points + np.array([drill_offset_x, drill_offset_y])

# Warp drill points with groove delta at closest groove fraction
warped_drill_points = []
drill_surface_z = []
for x0, y0 in centered_drill_points:
    # Find closest groove fraction to the projected distance on groove
    d_proj = gcode_path.project(ShapelyPoint(x0, y0))
    groove_frac_pt = d_proj / groove_len
    # Interpolate the deformation at that fraction
    delta = delta_interp(groove_frac_pt)
    # Offset in direction normal to tank center
    angle = np.arctan2(y0 - tank_center_y, x0 - tank_center_x)
    r_base = np.hypot(x0 - tank_center_x, y0 - tank_center_y)
    wx = tank_center_x + (r_base + delta) * np.cos(angle)
    wy = tank_center_y + (r_base + delta) * np.sin(angle)
    warped_drill_points.append([wx, wy])
    # Interpolate local surface Z for correct hole top
    z0 = z_interp(np.clip(d_proj, probe_proj_dist_sorted[0], probe_proj_dist_sorted[-1]))
    drill_surface_z.append(z0)
warped_drill_points = np.array(warped_drill_points)
drill_surface_z = np.array(drill_surface_z)

# ---- 7. Plot everything ----
plot.figure(figsize=(13, 13))
# Warped groove
points_g = np.array([smooth_groove_x, smooth_groove_y]).T.reshape(-1, 1, 2)
segments_g = np.concatenate([points_g[:-1], points_g[1:]], axis=1)
norm_g = plot.Normalize(np.min(surface_z-groove_depth), np.max(surface_z))
lc_g = LineCollection(segments_g, cmap='viridis', norm=norm_g)
final_groove_z = surface_z - groove_depth
lc_g.set_array(final_groove_z)
lc_g.set_linewidth(2)
plot.gca().add_collection(lc_g)

# Warped outcut
points_o = np.array([outcut_warped_x, outcut_warped_y]).T.reshape(-1, 1, 2)
segments_o = np.concatenate([points_o[:-1], points_o[1:]], axis=1)
norm_o = plot.Normalize(np.min(final_outcut_z), np.max(outcut_z))
lc_o = LineCollection(segments_o, cmap='plasma', norm=norm_o)
lc_o.set_array(final_outcut_z)
lc_o.set_linewidth(2)
plot.gca().add_collection(lc_o)

# Plot original (centered) and warped drill points for comparison
plot.scatter(centered_drill_points[:,0], centered_drill_points[:,1],
    marker='o', facecolors='none', edgecolors='r', s=120, linewidths=2, label='Original Drill Points (centered)', zorder=11)
plot.scatter(warped_drill_points[:,0], warped_drill_points[:,1],
    marker='x', color='black', s=120, linewidths=2, label='Warped Drill Points', zorder=12)

plot.plot(orig_groove_x, orig_groove_y, '--', color='gray', linewidth=1.2, label="Original Groove Path")
plot.plot(orig_outcut_x, orig_outcut_y, 'r--', linewidth=1.2, label="Original Outcut (dashed)")

plot.scatter(probed_points["x"], probed_points["y"], c=probed_points["z"], cmap='viridis', edgecolors='k', label='Probe Points', zorder=10, norm=norm_g)
plot.colorbar(lc_g, label="Groove Z (mm, deepest pass)", pad=0.01)
plot.colorbar(lc_o, label="Outcut Z (mm, deepest pass)", pad=0.04)
plot.gca().set_aspect('equal')
plot.xlabel("X (mm)")
plot.ylabel("Y (mm)")
plot.title(f"Groove and Outcut, Warped Depth\n(Color: final cut Z at each location)")
plot.legend()
plot.grid(True)
plot.tight_layout()
plot.show()

# ---- 8. Export G-code for groove, holes, outcut in a single file ----
spiral_stepdown = 2.0   # mm per spiral
hole_diameter = 8.5
tool_diameter = 7.0
hole_radius = hole_diameter / 2
tool_radius = tool_diameter / 2
offset = hole_radius - tool_radius   # 0.75mm offset from hole center

safe_height = 20.0
approach_height = 1.0
probe_offset_z = 44.0  # mm offset between probe zero and machine zero

# Calculate job travel height as 20mm above the highest point of the tank
highest_tank_point = np.max(surface_z)  # Highest Z value from probe data
job_travel_height = highest_tank_point + probe_offset_z + safe_height

with open("tank_full_job_warped.gcode", "w") as f:
    # --- Program header ---
    f.write("G21 ; mm units\nG90 ; absolute positioning\n$H ; Home\n")

    # ----------------------
    # --- 1. Warped Groove
    # ----------------------

    # Build pass depths: cut_step increments until groove_depth, with last pass exact
    groove_passes = list(np.arange(cut_step, groove_depth, cut_step))
    if groove_passes[-1] != groove_depth:
        groove_passes.append(groove_depth)

    for pass_depth in groove_passes:
        # Z at each path sample = local surface - pass_depth (parallel to surface)
        zpath = surface_z - pass_depth
        f.write(f"( Groove Pass {pass_depth}mm below local surface )\n")

        # Move to start point at safe Z
        f.write(f"G0 Z{job_travel_height:.3f}\n")
        f.write(f"G0 X{smooth_groove_x[0]:.3f} Y{smooth_groove_y[0]:.3f}\n")
        f.write(f"G1 Z{zpath[0] + probe_offset_z:.3f} F200\n")

        # Continuous loop following warped groove
        for xg, yg, zg in zip(smooth_groove_x[1:], smooth_groove_y[1:], zpath[1:]):
            f.write(f"G1 X{xg:.3f} Y{yg:.3f} Z{zg + probe_offset_z:.3f} F1000\n")

    # Retract after groove
    f.write(f"G0 Z{job_travel_height:.3f}\n")
    f.write("( End Groove )\n")

    # ----------------------
    # --- 2. Spiral Holes (no mid-lift)
    # ----------------------
    f.write("\n( Spiral-milled holes: helical 2mm per rev, no mid-lift )\n")
    for i, (x, y) in enumerate(warped_drill_points):
        z_top = float(drill_surface_z[i])           # local surface at this hole
        target_depth = z_top - outcut_depth         # final Z (more negative)
        start_x, start_y = x + offset, y            # start point on hole radius

        f.write(f"\n( Spiral Hole {i+1} at X{x:.3f} Y{y:.3f} )\n")
        # Safe approach
        f.write(f"G0 Z{job_travel_height:.3f}\n")
        f.write(f"G0 X{x:.3f} Y{y:.3f}\n")
        f.write(f"G0 Z{z_top + approach_height + probe_offset_z:.3f}\n")
        f.write(f"G0 X{start_x:.3f} Y{start_y:.3f}\n")

        # Helical spiral down: one full CCW circle per step to next depth
        current_depth = z_top
        while current_depth > target_depth + 1e-9:
            next_depth = max(current_depth - spiral_stepdown, target_depth)
            # G3 full circle, keeping XY at start point, descending to next_depth
            f.write(
                f"G3 X{start_x:.3f} Y{start_y:.3f} "
                f"Z{next_depth + probe_offset_z:.3f} "
                f"I{-offset:.3f} J0.000 F500\n"
            )
            current_depth = next_depth
    # Retract after the hole
    f.write(f"G0 Z{job_travel_height:.3f}\n")
    f.write("( End Holes )\n")

   
    # ----------------------
    # --- 3. Outcut
    # ----------------------
    lowest_probe_z = float(np.min(probed_points["z"]))
    final_plane_z  = lowest_probe_z - outcut_depth   # lowest surface point minus thickness

    # Build passes similar to groove: each path follows the warped surface, but stepped down
    def make_outcut_passes(surface_z, total_depth, step):
        passes = []
        depth = step
        while depth < total_depth - 1e-9:   # intermediate steps
            passes.append(surface_z - depth)
            depth += step
        # last pass goes exactly to final depth
        passes.append(surface_z - total_depth)
        return passes

    # Generate Z profiles for outcut (parallel to warped surface)
    outcut_passes = make_outcut_passes(outcut_z, outcut_depth, cut_step)


    f.write(f"( Outcut passes: lowest_probe_z {lowest_probe_z:.3f}, cut depth {outcut_depth:.3f}, final plane {final_plane_z:.3f} )\n")


    # Start positioning
    f.write(f"G0 Z{job_travel_height:.3f}\n")
    f.write(f"G0 X{outcut_warped_x[0]:.3f} Y{outcut_warped_y[0]:.3f}\n")

    # Loop over passes
    for pass_num, zpath in enumerate(outcut_passes, 1):
        f.write(f"\n( Outcut pass {pass_num}: {cut_step if pass_num < len(outcut_passes) else outcut_depth - cut_step*(len(outcut_passes)-1):.3f}mm step )\n")
        # move down to start Z
        f.write(f"G1 Z{zpath[0] + probe_offset_z:.3f} F200\n")

        # Follow warped surface with local Z adjustments
        for xo, yo, zo in zip(outcut_warped_x[1:], outcut_warped_y[1:], zpath[1:]):
            f.write(f"G1 X{xo:.3f} Y{yo:.3f} Z{zo + probe_offset_z:.3f} F1000\n")

    # Retract at the end
    f.write(f"G0 Z{job_travel_height:.3f}\n")
    f.write("\nM2 ; End of program\n")

