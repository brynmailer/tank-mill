import pandas as pd
import numpy as np

def load_points(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    
    # Initialize lists to store processed coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Process pairs of rows
    for i in range(0, len(raw) - 1, 2):
        try:
            # Get x,y from current row (even index)
            x = raw.iloc[i]['x']
            y = raw.iloc[i]['y']
            
            # Get z from next row (odd index)
            z = raw.iloc[i + 1]['z']
            
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Skipping invalid data at rows {i}-{i+1}: {e}")
            continue
    
    # Create and return DataFrame
    return pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'z': z_coords
    })

def arc_g2(
    start_x, start_y,
    end_x, end_y,
    offset_x, offset_y,
    steps=100):
    center_x, center_y = start_x + offset_x, start_y + offset_y
    radius = np.hypot(offset_x, offset_y)

    a0 = np.arctan2(start_y - center_y, start_x - center_x)
    a1 = np.arctan2(end_y - center_y, end_x - center_x)

    if a1 > a0:
        a1 -= 2 * np.pi

    angles = np.linspace(a0, a1, steps)
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)
    xs[-1], ys[-1] = end_x, end_y

    return xs, ys

def arc_g3(
        start_x, start_y,
        end_x, end_y,
        offset_x, offset_y,
        steps=100):
    center_x, center_y = start_x + offset_x, start_y + offset_y
    radius = np.hypot(offset_x, offset_y)

    a0 = np.arctan2(start_y - center_y, start_x - center_x)
    a1 = np.arctan2(end_y - center_y, end_x - center_x)

    if a1 <= a0:
        a1 += 2 * np.pi

    angles = np.linspace(a0, a1, steps)
    xs = center_x + radius * np.cos(angles)
    ys = center_y + radius * np.sin(angles)
    xs[-1], ys[-1] = end_x, end_y

    return xs, ys

def build_path(segments, arc_func, steps=100):
    path_x, path_y = [], []

    for (x0, y0), (x1, y1), I, J in segments:
        if I is None:
            xs = np.linspace(x0, x1, steps)
            ys = np.linspace(y0, y1, steps)
        else:
            xs, ys = arc_func(x0, y0, x1, y1, I, J, steps)

        if len(path_x) > 0:
            xs, ys = xs[1:], ys[1:]

        path_x.extend(xs)
        path_y.extend(ys)

    return np.array(path_x), np.array(path_y)
