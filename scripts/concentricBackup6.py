import numpy as np
import csv
import matplotlib.pyplot as plt

# PARAMETERS
n = 79      # inward offset in mm for inner points
mm = 64     # inward offset in mm for expected points
x = [4, 2, 5, 3, 3, 3, 4]   # number of equidistant points per edge (list)
y = 109     # outward perpendicular offset (for outer points)
z_offsets = [1, 30, 30, 30, 30, 30, 30]  # one z-offset PER CORNER (must match number of vertices)
circle_radius = 64.0

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

def flip_points(points): return [(-px-616,py) for (px,py) in points]

# ---------- Validate ----------
if len(x)!=len(edges) or len(z_offsets)!=len(edges):
    raise ValueError("x and z_offsets must match number of edges")

# ---------- Build offset edge sets ----------
inner_edges, expected_edges = generate_inner_edges(edges,n), generate_inner_edges(edges,mm)

# ---------- Generate points ----------
inner_points, inner_offs = [], []
expected_points, expected_offs = [], []
outer_points, outer_offs = [], []
circle_points = []

for i,((inner_p1,inner_p2),(exp_p1,exp_p2)) in enumerate(zip(inner_edges,expected_edges)):
    num = x[i]
    inner_eq = equidistant_points_including_endpoints(inner_p1,inner_p2,num)
    exp_eq   = equidistant_points_including_endpoints(exp_p1,exp_p2,num)
    inner_points.extend(inner_eq); expected_points.extend(exp_eq)

    # circles only on middle expected points
    if num>2: circle_points.extend(exp_eq[1:-1])

    # z offsets: current corner (i) at start, next corner (i+1) at end
    z1, z2 = z_offsets[i], z_offsets[(i+1)%len(edges)]
    inner_off = offset_endpoints_along_edge(inner_p1,inner_p2,z1,z2)
    exp_off   = offset_endpoints_along_edge(exp_p1,exp_p2,z1,z2)
    inner_offs.extend(inner_off); expected_offs.extend(exp_off)
    circle_points.extend(exp_off)  # z-offset endpoints also get circles

    outer_points.extend(offset_points_outward(inner_eq,inner_p1,inner_p2,y))
    outer_offs.extend(offset_points_outward(inner_off,inner_p1,inner_p2,y))

# Combine
inner_all, expected_all, outer_all = inner_points+inner_offs, expected_points+expected_offs, outer_points+outer_offs
circle_centers, circle_centers_m = circle_points, flip_points(circle_points)

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
plot_points(ax1,inner_points,'blue',"A Inner eq pts")
plot_points(ax1,expected_points,'green',"A Expected eq pts")
plot_points(ax1,outer_points,'red',"A Outer eq pts")
plot_points(ax1,inner_offs,'blue',"A Inner endpoint+z",marker='x',size=60)
plot_points(ax1,expected_offs,'green',"A Expected endpoint+z",marker='x',size=60)
plot_points(ax1,outer_offs,'red',"A Outer endpoint+z",marker='x',size=60)
plot_circles(ax1,circle_centers,circle_radius,'green')
ax1.legend(); ax1.grid(True)
ax1.set_title("A Data: Circles on middle expected pts + z-offset endpoints")

# B plot (mirrored)
edges_m=[tuple(flip_points(list(e))) for e in edges]
inner_edges_m=[tuple(flip_points(list(e))) for e in inner_edges]
expected_edges_m=[tuple(flip_points(list(e))) for e in expected_edges]
plot_edges(ax2,edges_m,'gray','Original (mirrored)')
plot_edges(ax2,inner_edges_m,'blue',f'Inner Offset ({n}mm)')
plot_edges(ax2,expected_edges_m,'green',f'Expected Offset ({mm}mm)')
plot_points(ax2,flip_points(inner_points),'blue',"B Inner eq pts")
plot_points(ax2,flip_points(expected_points),'green',"B Expected eq pts")
plot_points(ax2,flip_points(outer_points),'red',"B Outer eq pts")
plot_points(ax2,flip_points(inner_offs),'blue',"B Inner endpoint+z",marker='x',size=60)
plot_points(ax2,flip_points(expected_offs),'green',"B Expected endpoint+z",marker='x',size=60)
plot_points(ax2,flip_points(outer_offs),'red',"B Outer endpoint+z",marker='x',size=60)
plot_circles(ax2,circle_centers_m,circle_radius,'green')
ax2.legend(); ax2.grid(True)
ax2.set_title("B Data (mirrored): Circles on middle expected pts + z-offset endpoints")

plt.tight_layout(); plt.show()
