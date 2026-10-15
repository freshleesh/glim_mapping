#!/usr/bin/env python3
"""Build a ROS map_server 2D occupancy grid (PGM+YAML) from a GLIM merged cloud.
Occupied = obstacle-band points; free = drivable area flood-filled from the trajectory.
Usage: make_occgrid.py <merged.ply> <traj_lidar.txt> <out_prefix> [res] [zmin_off] [zmax_off]
"""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_map import read_ply

ply, traj_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
res = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
zmin_off = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15   # above floor
zmax_off = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5

MIN_PTS = 4  # cell needs >= this many points to count as a solid obstacle (reject noise)
data, _ = read_ply(ply)
xyz = data[:, :3]
traj = np.loadtxt(traj_p)[:, 1:4]
# obstacle band centered on the lidar/trajectory scanning height (thin -> clean walls)
sensor_z = np.median(traj[:, 2])
obst = xyz[(xyz[:, 2] > sensor_z - 0.30) & (xyz[:, 2] < sensor_z + 0.30)]

mn = xyz[:, :2].min(0) - 0.5
mx = xyz[:, :2].max(0) + 0.5
W = int((mx[0] - mn[0]) / res) + 1
H = int((mx[1] - mn[1]) / res) + 1

def to_cell(p):
    gx = ((p[:, 0] - mn[0]) / res).astype(int)
    gy = ((p[:, 1] - mn[1]) / res).astype(int)
    return gx, gy

count = np.zeros((H, W), np.int32)
gx, gy = to_cell(obst)
ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
np.add.at(count, (gy[ok], gx[ok]), 1)
occ = count >= MIN_PTS
# close small gaps so flood-fill can't leak through sparse walls/doorways
from scipy.ndimage import binary_dilation, binary_closing
occ_barrier = binary_closing(binary_dilation(occ, iterations=2), structure=np.ones((3,3)), iterations=2)

# flood fill free space from trajectory cells, blocked by occupied
from collections import deque
free = np.zeros((H, W), bool)
seeds = deque()
tgx, tgy = to_cell(traj)
for x, y in zip(tgx, tgy):
    if 0 <= x < W and 0 <= y < H and not occ[y, x]:
        seeds.append((y, x))
seen = np.zeros((H, W), bool)
while seeds:
    y, x = seeds.popleft()
    if seen[y, x] or occ_barrier[y, x]:
        continue
    seen[y, x] = True; free[y, x] = True
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
        ny, nx = y+dy, x+dx
        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and not occ_barrier[ny, nx]:
            seeds.append((ny, nx))
# only keep reachable free; occupied stays the (un-dilated) true wall points

# map_server PGM: 0=occupied(black), 254=free(white), 205=unknown
img = np.full((H, W), 205, np.uint8)
img[free] = 254
img[occ] = 0
# PGM origin is top-left; ROS map y increases upward -> flip rows
img_flipped = np.flipud(img)

with open(out + '.pgm', 'wb') as f:
    f.write(b'P5\n%d %d\n255\n' % (W, H))
    f.write(img_flipped.tobytes())

# origin = world coord of lower-left pixel
with open(out + '.yaml', 'w') as f:
    f.write(f"image: {out.split('/')[-1]}.pgm\n")
    f.write(f"resolution: {res}\n")
    f.write(f"origin: [{mn[0]:.4f}, {mn[1]:.4f}, 0.0]\n")
    f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

print(f"grid {W}x{H} @ {res}m  occ={occ.sum()} free={free.sum()}  -> {out}.pgm/.yaml")
