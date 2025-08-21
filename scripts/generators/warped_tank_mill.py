import sys
sys.path.append("./src")

import matplotlib.pyplot as plot
plot.close('all') # close any existing figures from previous runs
import numpy as np
import json
from shapely.geometry import LineString, Point as ShapelyPoint
from scipy.interpolate import splprep, splev, interp1d, Rbf
from matplotlib.collections import LineCollection

import utils 

# ---- Settings ----
path = "./data/b_points.csv"
#path = "expected_a_probe_points.csv"
groove_depth = 5     # Groove cut depth (mm)
outcut_depth = 10    # Outcut depth (mm)
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

def ensure_closed_path(x, y, tolerance=0.1):
    """Force the path to close exactly by making the last point equal to the first"""
    x = np.array(x)
    y = np.array(y)
    
    # Check if already closed within tolerance
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) > tolerance:
        print(f"Path gap: {np.hypot(x[-1] - x[0], y[-1] - y[0]):.3f}mm - forcing closure")
    
    # Force exact closure
    x[-1] = x[0]
    y[-1] = y[0]
    return x, y

def validate_path_closure(x, y, name="Path"):
    """Check and report path closure quality"""
    gap = np.hypot(x[-1] - x[0], y[-1] - y[0])
    print(f"{name} closure gap: {gap:.6f} mm")
    if gap > 0.01:  # 0.01mm tolerance
        print(f"WARNING: {name} may not be properly closed!")
    return gap < 0.01


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
    return d - baseline  # positive = push outward, negative = pull inward

rbf = Rbf(
    probed_points["x"], probed_points["y"],
    [residual(row["x"], row["y"]) for _, row in probed_points.iterrows()],
    function='linear'
)

# Warp groove points using local normal
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

# Smooth warped groove path
tck, u = splprep([warped_x, warped_y], s=5.0, per=True)
unew = np.linspace(0, 1.0, 1000)
smooth_groove_x, smooth_groove_y = splev(unew, tck)

# Force exact closure of the groove path
smooth_groove_x, smooth_groove_y = ensure_closed_path(smooth_groove_x, smooth_groove_y)
validate_path_closure(smooth_groove_x, smooth_groove_y, "Groove")

gcode_path = LineString(list(zip(smooth_groove_x, smooth_groove_y)))

# ---- Depth interpolation along groove (fixed: 1D, periodic, no X warping) ----
from scipy.signal import savgol_filter

# Project probe points onto the (already smoothed) groove to get their s-positions
probe_proj_dist = [gcode_path.project(ShapelyPoint(row["x"], row["y"]))
                   for _, row in probed_points.iterrows()]
probe_proj_dist = np.asarray(probe_proj_dist, dtype=float)

# Sort probes by distance along the groove so interpolation is monotonic
sort_idx = np.argsort(probe_proj_dist)
probe_proj_dist_sorted = probe_proj_dist[sort_idx]
z_values_sorted = np.asarray(probed_points["z"], dtype=float)[sort_idx]

# Periodic extension of probe samples to avoid edge effects at the seam
groove_length = gcode_path.length
extended_s = np.concatenate([
    probe_proj_dist_sorted - groove_length,
    probe_proj_dist_sorted,
    probe_proj_dist_sorted + groove_length
])
extended_z = np.concatenate([z_values_sorted, z_values_sorted, z_values_sorted])

# Build a simple *1D* linear interpolator on the extended (periodic) data
z_interp = interp1d(extended_s, extended_z, kind="linear",
                    bounds_error=False, fill_value="extrapolate")

# Sample Z on a strictly monotonic arc-length grid (no duplicate endpoint)
n_groove = len(smooth_groove_x)
arc_s = np.linspace(0.0, groove_length, n_groove, endpoint=False)
surface_z_raw = z_interp(arc_s)

# Optional gentle smoothing of Z ONLY (periodic Savitzky–Golay)
# Window ≈ 2% of path samples, forced odd and at least 7
win = max(7, 2 * max(1, n_groove // 50) + 1)
surface_z = savgol_filter(surface_z_raw, window_length=win, polyorder=3, mode="wrap")

# Ensure exact closure numerically by defining the last point implicit (endpoint=False).
# If you prefer an explicit last sample for plotting, you can append arc_s[0], surface_z[0].
# ------------------------------------------------------------------------------- end fix


# STEP 5: Add the improved G-code generation function after line 275 (after feed definitions)
# ADD AFTER LINE 275 (after rapid_feed = 3000.0):

def write_closed_groove_passes(f, x_path, y_path, z_passes,
                               probe_offset_z, feed_plunge, feed_linear,
                               job_travel_height, groove_depth, cut_step):
    """
    Emit groove passes as closed loops, staying down between passes.
    We rapid to safe and XY start ONCE, then just plunge deeper for each pass.
    """

    x_path = np.asarray(x_path, dtype=float)
    y_path = np.asarray(y_path, dtype=float)

    # Go to safe and XY start ONCE
    write_rapid(f, z=job_travel_height)
    write_rapid(f, x=x_path[0], y=y_path[0])

    for idx, zpath in enumerate(z_passes, 1):
        zpath = np.asarray(zpath, dtype=float)
        # ensure exact Z closure for this pass
        zpath[-1] = zpath[0]

        # Plunge to this pass start Z (no retract between passes)
        f.write(f"G1 Z{zpath[0] + probe_offset_z:.3f} F{feed_plunge:.0f}\n")

        # Cut around the closed path
        for i in range(1, len(x_path)):
            f.write(
                f"G1 X{x_path[i]:.3f} Y{y_path[i]:.3f} "
                f"Z{zpath[i] + probe_offset_z:.3f} F{feed_linear:.0f}\n"
            )

        # Explicitly close back to exact start (same Z) to avoid any seam
        f.write(
            f"G1 X{x_path[0]:.3f} Y{y_path[0]:.3f} "
            f"Z{zpath[0] + probe_offset_z:.3f} F{feed_linear:.0f}\n"
        )

    # Do NOT retract here; your caller already does:
    # write_rapid(f, z=job_travel_height)



# ---- Build deformation mapping along groove ----
groove_len = gcode_path.length
n_samples = 1000
groove_distances = np.linspace(0, groove_len, n_samples)
groove_pts = np.array([gcode_path.interpolate(d) for d in groove_distances])
groove_frac = groove_distances / groove_len

# Deformation delta at sampled groove points
groove_delta = np.array([rbf(pt.x, pt.y) for pt in groove_pts])

# Interpolator: fraction of groove → deformation delta
delta_interp = interp1d(
    groove_frac,
    groove_delta,
    kind='linear',
    fill_value='extrapolate'
)

# ---- 5. Fixed Outcut Warping - Direct Normal Displacement Method ----
from scipy.interpolate import interp1d

# The key insight: Apply the groove's deformation to the outcut by using the groove's 
# displacement field and applying it in the direction from groove toward outcut

# Step 1: Build the groove's deformation field
groove_path_obj = LineString(list(zip(smooth_groove_x, smooth_groove_y)))
groove_len = groove_path_obj.length

# Sample points along the groove for deformation analysis
n_samples = 1000
groove_distances = np.linspace(0, groove_len, n_samples, endpoint=False)
groove_fractions = groove_distances / groove_len

# Get warped groove points
warped_groove_pts = [groove_path_obj.interpolate(d) for d in groove_distances]
warped_gx = np.array([p.x for p in warped_groove_pts])
warped_gy = np.array([p.y for p in warped_groove_pts])

# Get corresponding original groove points at same fractions
orig_groove_path = LineString(list(zip(orig_groove_x, orig_groove_y)))
orig_groove_len = orig_groove_path.length
orig_distances = groove_fractions * orig_groove_len
orig_groove_pts = [orig_groove_path.interpolate(d) for d in orig_distances]
orig_gx = np.array([p.x for p in orig_groove_pts])
orig_gy = np.array([p.y for p in orig_groove_pts])

# Calculate displacement vectors
disp_x = warped_gx - orig_gx
disp_y = warped_gy - orig_gy

# Create periodic interpolators for the displacement
extended_fractions = np.concatenate([groove_fractions - 1.0, groove_fractions, groove_fractions + 1.0])
extended_disp_x = np.concatenate([disp_x, disp_x, disp_x])
extended_disp_y = np.concatenate([disp_y, disp_y, disp_y])

disp_x_interp = interp1d(extended_fractions, extended_disp_x, kind='linear', bounds_error=False, fill_value='extrapolate')
disp_y_interp = interp1d(extended_fractions, extended_disp_y, kind='linear', bounds_error=False, fill_value='extrapolate')

# Step 2: Apply displacement to outcut
outcut_path_obj = LineString(list(zip(outcut_x, outcut_y)))
outcut_len = outcut_path_obj.length

# Sample outcut points
outcut_distances = np.linspace(0, outcut_len, n_samples, endpoint=False)
outcut_pts = [outcut_path_obj.interpolate(d) for d in outcut_distances]
outcut_x_orig = np.array([p.x for p in outcut_pts])
outcut_y_orig = np.array([p.y for p in outcut_pts])

# Method: For each outcut point, find the nearest groove point and apply that displacement
outcut_warped_x = np.zeros(n_samples)
outcut_warped_y = np.zeros(n_samples)

for i, (ox, oy) in enumerate(zip(outcut_x_orig, outcut_y_orig)):
    # Find the closest point on the groove path
    closest_dist_on_groove = groove_path_obj.project(ShapelyPoint(ox, oy))
    closest_groove_pt = groove_path_obj.interpolate(closest_dist_on_groove)
    
    # Get the groove fraction for this closest point
    groove_frac = closest_dist_on_groove / groove_len
    
    # Get displacement at this groove fraction
    dx = disp_x_interp(groove_frac)
    dy = disp_y_interp(groove_frac)
    
    # Apply the same displacement to the outcut point
    outcut_warped_x[i] = ox + dx
    outcut_warped_y[i] = oy + dy

# Alternative method if the above doesn't work well:
# Use radial displacement from tank center
use_radial_displacement = True

if use_radial_displacement:
    # Get tank center
    tank_center_x = (probed_points["x"].min() + probed_points["x"].max()) / 2
    tank_center_y = (probed_points["y"].min() + probed_points["y"].max()) / 2
    
    # Calculate radial displacement at each groove point
    radial_displacements = []
    for i in range(n_samples):
        # Original radius from center
        orig_radius = np.hypot(orig_gx[i] - tank_center_x, orig_gy[i] - tank_center_y)
        # Warped radius from center  
        warped_radius = np.hypot(warped_gx[i] - tank_center_x, warped_gy[i] - tank_center_y)
        # Radial displacement (positive = outward)
        radial_displacements.append(warped_radius - orig_radius)
    
    radial_displacements = np.array(radial_displacements)
    
    # Create radial displacement interpolator
    extended_radial_disp = np.concatenate([radial_displacements, radial_displacements, radial_displacements])
    radial_disp_interp = interp1d(extended_fractions, extended_radial_disp, 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
    
    # Apply radial displacement to outcut
    for i, (ox, oy) in enumerate(zip(outcut_x_orig, outcut_y_orig)):
        # Find corresponding groove fraction
        closest_dist_on_groove = groove_path_obj.project(ShapelyPoint(ox, oy))
        groove_frac = closest_dist_on_groove / groove_len
        
        # Get radial displacement
        radial_disp = radial_disp_interp(groove_frac)
        
        # Apply displacement radially from tank center
        angle = np.arctan2(oy - tank_center_y, ox - tank_center_x)
        current_radius = np.hypot(ox - tank_center_x, oy - tank_center_y)
        new_radius = current_radius + radial_disp
        
        outcut_warped_x[i] = tank_center_x + new_radius * np.cos(angle)
        outcut_warped_y[i] = tank_center_y + new_radius * np.sin(angle)

# Clean the warped path
valid_mask = np.isfinite(outcut_warped_x) & np.isfinite(outcut_warped_y)
outcut_warped_x = outcut_warped_x[valid_mask]
outcut_warped_y = outcut_warped_y[valid_mask]

# Safety check - if we lost too many points, fall back to original
if len(outcut_warped_x) < n_samples * 0.8:  # Keep at least 80% of points
    print("Warning: Lost too many points in outcut warping, using original path")
    outcut_warped_x = outcut_x
    outcut_warped_y = outcut_y

# Calculate Z values for warped outcut
if len(outcut_warped_x) > 0:
    outcut_path_obj_warped = LineString(list(zip(outcut_warped_x, outcut_warped_y)))
    outcut_proj_dist = []
    for x, y in zip(outcut_warped_x, outcut_warped_y):
        try:
            dist = groove_path_obj.project(ShapelyPoint(x, y))
            outcut_proj_dist.append(dist)
        except:
            # If projection fails, use closest existing distance
            outcut_proj_dist.append(outcut_proj_dist[-1] if outcut_proj_dist else 0.0)
    
    outcut_proj_dist = np.array(outcut_proj_dist)
    outcut_z = z_interp(np.clip(outcut_proj_dist, probe_proj_dist_sorted[0], probe_proj_dist_sorted[-1]))
    final_outcut_z = outcut_z - outcut_depth
else:
    # Fallback if warping completely failed
    final_outcut_z = np.full(len(outcut_x), np.mean(probed_points["z"]) - outcut_depth)


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

    # Find the closest point on the groove
    closest_pt = gcode_path.interpolate(d_proj)
    idx_closest = np.argmin([closest_pt.distance(ShapelyPoint(gx, gy)) for gx, gy in zip(smooth_groove_x, smooth_groove_y)])

    # Local tangent & normal at that groove point
    if 0 < idx_closest < len(smooth_groove_x) - 1:
        dx = smooth_groove_x[idx_closest + 1] - smooth_groove_x[idx_closest - 1]
        dy = smooth_groove_y[idx_closest + 1] - smooth_groove_y[idx_closest - 1]
    else:
        dx, dy = 1.0, 0.0  # fallback
    tangent = np.array([dx, dy])
    tangent /= np.linalg.norm(tangent)
    nx, ny = -tangent[1], tangent[0]

    # Warp drill point along the same normal direction as groove
    wx = x0 + delta * nx
    wy = y0 + delta * ny

    warped_drill_points.append([wx, wy])
    # Interpolate local surface Z for correct hole top
    z0 = z_interp(np.clip(d_proj, probe_proj_dist_sorted[0], probe_proj_dist_sorted[-1]))
    drill_surface_z.append(z0)
warped_drill_points = np.array(warped_drill_points)
drill_surface_z = np.array(drill_surface_z)


# ---- 7. Export G-code for groove, holes, outcut in a single file ----
spiral_stepdown = 2.0   # mm per spiral
hole_diameter   = 8.5
tool_diameter   = 7.0
hole_radius     = hole_diameter / 2
tool_radius     = tool_diameter / 2
offset          = hole_radius - tool_radius   # 0.75mm offset from hole center

safe_height     = 20.0
approach_height = 1.0
probe_offset_z  = 44.0  # mm offset between probe zero and machine zero
park_x, park_y = -10.0, 1200.0


# ---- Feeds (mm/min) ----
feed_plunge = 200.0     # Z-only plunges / re-plunges
feed_linear = 1000.0    # cutting moves (G1 XY/XYZ)
feed_arc    = 700.0     # helical/circular arcs (G2/G3)

# ---- “Rapid” between cuts ----
# Many controllers don't let you change G0 speed. If you want a controllable
# "rapid", set use_g1_rapid=True and we’ll move with G1 at rapid_feed.
use_g1_rapid = False
rapid_feed   = 3000.0   # only used when use_g1_rapid=True

def write_rapid(fh, x=None, y=None, z=None):
    """Emit a rapid move. Uses G0 by default, or G1 with rapid_feed if configured."""
    parts = []
    if x is not None: parts.append(f"X{x:.3f}")
    if y is not None: parts.append(f"Y{y:.3f}")
    if z is not None: parts.append(f"Z{z:.3f}")
    if not parts:
        return
    if use_g1_rapid:
        fh.write("G1 " + " ".join(parts) + f" F{rapid_feed:.0f}\n")
    else:
        fh.write("G0 " + " ".join(parts) + "\n")

# Calculate job travel height as 20mm above the highest point of the tank
highest_tank_point = float(np.max(surface_z))
job_travel_height  = highest_tank_point + probe_offset_z + safe_height

# Utility: build “parallel-to-surface” Z paths for a given total depth & step
def make_parallel_passes(surface_array, total_depth, step):
    """Returns list of z-path arrays: surface_array - depth_i, last pass exact."""
    passes = []
    depth = float(step)
    while depth < total_depth - 1e-9:
        passes.append(surface_array - depth)
        depth += step
    passes.append(surface_array - float(total_depth))  # exact final pass
    return passes

with open("tank_full_job_warped.gcode", "w") as f:
    # --- Program header ---
    # f.write("G21 ; mm units\nG90 ; absolute positioning\n$H ; Home\n")


    # ----------------------
    # --- 1) GROOVE
    # ----------------------
    groove_passes = make_parallel_passes(surface_z, groove_depth, cut_step)
    write_closed_groove_passes(f, smooth_groove_x, smooth_groove_y, groove_passes, 
                              probe_offset_z, feed_plunge, feed_linear, job_travel_height, 
                              groove_depth, cut_step)

    write_rapid(f, z=job_travel_height)
    f.write("( End Groove )\n")

    # ----------------------
    # --- 2) SPIRAL HOLES (no mid-lift)
    # ----------------------
    f.write("\n( Spiral-milled holes: helical 2mm per rev, no mid-lift )\n")
    for i, (x, y) in enumerate(warped_drill_points, 1):
        z_top        = float(drill_surface_z[i-1])     # local surface
        target_depth = z_top - outcut_depth            # final Z
        start_x, start_y = x + offset, y

        f.write(f"\n( Spiral Hole {i} at X{x:.3f} Y{y:.3f} )\n")
        write_rapid(f, z=job_travel_height)
        write_rapid(f, x=x, y=y)
        write_rapid(f, z=z_top + approach_height + probe_offset_z)
        write_rapid(f, x=start_x, y=start_y)

        current_depth = z_top
        while current_depth > target_depth + 1e-9:
            next_depth = max(current_depth - spiral_stepdown, target_depth)
            f.write(
                f"G3 X{start_x:.3f} Y{start_y:.3f} "
                f"Z{next_depth + probe_offset_z:.3f} "
                f"I{-offset:.3f} J0.000 F{feed_arc:.0f}\n"
            )
            current_depth = next_depth

        write_rapid(f, z=job_travel_height)
    f.write("( End Holes )\n")

    # ----------------------
    # --- 3) OUTCUT (parallel passes, final pass constant at lowest Z − thickness)
    # ----------------------

    # Build parallel depths: cut_step increments up to (but not including) the final depth
    depths = list(np.arange(cut_step, outcut_depth, cut_step))  # e.g. 3, 6, 9 for 12

    # Parallel passes (each follows warped surface)
    parallel_passes = [outcut_z - d for d in depths]

    # Final constant plane: lowest probed point minus tank thickness (uniform around path)
    lowest_probe_z = float(np.min(probed_points["z"]))
    final_plane_z  = lowest_probe_z - outcut_depth
    last_pass      = np.full_like(outcut_z, final_plane_z)

    # Combine: all parallel passes, then the constant-depth finishing pass
    outcut_passes = parallel_passes + [last_pass]

    
    f.write(f"\n( Outcut: final plane = {final_plane_z:.3f}, "
        f"lowest probe {lowest_probe_z:.3f}, depth {outcut_depth:.3f} )\n")

    # Go to safe Z and XY start once
    write_rapid(f, z=job_travel_height)
    write_rapid(f, x=outcut_warped_x[0], y=outcut_warped_y[0])

    # Start spindle @ 12000 RPM after arriving at start
    f.write("(Spindle ON)\nM3 S12000\nG4 P2 ; 2s spin-up\n")

    for idx, zpath in enumerate(outcut_passes, 1):
        is_last = (idx == len(outcut_passes))
        step_desc = (cut_step if not is_last else (outcut_depth - cut_step * len(depths)))
        f.write(f"\n( Outcut pass {idx}: {'final constant plane' if is_last else f'{step_desc:.3f}mm step, parallel to surface'} )\n")

        # Plunge to the pass start Z (add probe_offset_z when writing)
        f.write(f"G1 Z{zpath[0] + probe_offset_z:.3f} F{feed_plunge:.0f}\n")

        # Follow the path: for parallel passes Z varies with XY; for last pass it's a constant array
        for xo, yo, zo in zip(outcut_warped_x[1:], outcut_warped_y[1:], zpath[1:]):
            f.write(f"G1 X{xo:.3f} Y{yo:.3f} Z{zo + probe_offset_z:.3f} F{feed_linear:.0f}\n")

    
    write_rapid(f, z=job_travel_height) # Retract after outcut
    f.write("\n( Park at end )\n")      # Park at requested position
    write_rapid(f, x=park_x, y=park_y)
    f.write("( Stop spindle )\nM5\n")   # Stop spindle


# ---- 8. Plot everything ----
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

fig = plot.figure(figsize=(13, 13))
ax = plot.gca()

# Warped groove
points_g = np.array([smooth_groove_x, smooth_groove_y]).T.reshape(-1, 1, 2)
segments_g = np.concatenate([points_g[:-1], points_g[1:]], axis=1)
norm_g = plot.Normalize(np.min(surface_z - groove_depth), np.max(surface_z))
lc_g = LineCollection(segments_g, cmap='viridis', norm=norm_g)
final_groove_z = surface_z - groove_depth
lc_g.set_array(final_groove_z)
lc_g.set_linewidth(2)
ax.add_collection(lc_g)

# Warped outcut
points_o = np.array([outcut_warped_x, outcut_warped_y]).T.reshape(-1, 1, 2)
segments_o = np.concatenate([points_o[:-1], points_o[1:]], axis=1)
norm_o = plot.Normalize(np.min(final_outcut_z), np.max(outcut_z))
lc_o = LineCollection(segments_o, cmap='plasma', norm=norm_o)
lc_o.set_array(final_outcut_z)
lc_o.set_linewidth(2)
ax.add_collection(lc_o)

# Circles around probed points
probe_circle_radius_big = 99.25
probe_circle_radius_small = 64.0

# Blue 99.25 mm
circles_big = [Circle((float(x), float(y)), radius=probe_circle_radius_big)
               for x, y in zip(probed_points["x"], probed_points["y"])]
pc_big = PatchCollection(circles_big, facecolor='none', edgecolor='tab:blue',
                         linewidth=1.2, linestyle='--', alpha=0.9, zorder=9)
ax.add_collection(pc_big)

# Red 64 mm
circles_small = [Circle((float(x), float(y)), radius=probe_circle_radius_small)
                 for x, y in zip(probed_points["x"], probed_points["y"])]
pc_small = PatchCollection(circles_small, facecolor='none', edgecolor='red',
                           linewidth=1.2, linestyle='--', alpha=0.9, zorder=9)
ax.add_collection(pc_small)

# Plot original (centered) and warped drill points
plot.scatter(centered_drill_points[:,0], centered_drill_points[:,1],
             marker='o', facecolors='none', edgecolors='r', s=120, linewidths=2,
             label='Original Drill Points (centered)', zorder=11)
plot.scatter(warped_drill_points[:,0], warped_drill_points[:,1],
             marker='x', color='black', s=120, linewidths=2,
             label='Warped Drill Points', zorder=12)

plot.plot(orig_groove_x, orig_groove_y, '--', color='gray', linewidth=1.2,
          label="Original Groove Path")
plot.plot(orig_outcut_x, orig_outcut_y, 'r--', linewidth=1.2,
          label="Original Outcut (dashed)")

# Probe points (colored by groove final Z)
plot.scatter(probed_points["x"], probed_points["y"],
             c=probed_points["z"], cmap='viridis', edgecolors='k',
             label='Probe Points', zorder=10, norm=norm_g)

# Colorbars
plot.colorbar(lc_g, label="Groove Z (mm, deepest pass)", pad=0.01)
plot.colorbar(lc_o, label="Outcut Z (mm, deepest pass)", pad=0.04)

# Axes / labels
ax.set_aspect('equal')
plot.xlabel("X (mm)")
plot.ylabel("Y (mm)")
plot.title("Groove and Outcut, Warped Depth\n(Color: final cut Z at each location)")

# Legend including circle styles
from matplotlib.lines import Line2D
circle_key_big = Line2D([0], [0], color='tab:blue', lw=1.2, ls='--', label='99.25 mm radius')
circle_key_small = Line2D([0], [0], color='red', lw=1.2, ls='--', label='64.0 mm radius')
handles, labels = ax.get_legend_handles_labels()
handles.extend([circle_key_big, circle_key_small])
labels.extend(['99.25 mm radius', '64.0 mm radius'])
ax.legend(handles, labels)

plot.grid(True)
plot.tight_layout()
plot.show()


# ---- Z Profile vs Path Length ----
fig2, ax2 = plot.subplots(figsize=(12, 6))

# Path distances along groove (in meters)
path_dist = np.linspace(0, groove_len, len(surface_z)) / 1000.0  # convert mm → m

# Expected flat Z line (-70 mm baseline)
ax2.axhline(-70, color="red", linestyle="--", linewidth=2, label="Expected Z = -70 mm")

# Probed points (plotted at their projected distance on groove)
probe_dist = np.array([gcode_path.project(ShapelyPoint(x, y)) for x, y in probed_points[["x","y"]].values]) / 1000.0
ax2.scatter(probe_dist, probed_points["z"], color="black", marker="o", label="Probed Points")

# Groove first pass (tank top approximation)
first_pass_surface = groove_passes[0] + cut_step
ax2.plot(path_dist, first_pass_surface, "-", color="blue", linewidth=2, label="Groove First Pass (tank top)")

# Groove last pass (deepest cut)
last_pass_surface = groove_passes[-1]
ax2.plot(path_dist, last_pass_surface, "-", color="green", linewidth=2, label="Groove Last Pass (deepest cut)")

# Force Z axis tick spacing to 1 mm
import matplotlib.ticker as mticker
ax2.yaxis.set_major_locator(mticker.MultipleLocator(1))

# Labels / styling
ax2.set_xlabel("Path Distance (m)")
ax2.set_ylabel("Z Height (mm)")
ax2.set_title("Tank Surface & Groove Cut Depths vs Path Length")
ax2.legend()
ax2.grid(True)

plot.tight_layout()
plot.show()
