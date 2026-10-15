#!/usr/bin/env python3
"""Evaluate a GLIM merged map (binary PLY) for F1TENTH precision.
Outputs metrics.json + top-down / side render PNGs.
Usage: eval_map.py <merged.ply> <out_dir> [traj_lidar.txt]
"""
import sys, json, struct
import numpy as np

def read_ply(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'ply'
        fmt = f.readline().strip()
        n = 0
        props = []
        while True:
            line = f.readline().strip()
            if line.startswith(b'element vertex'):
                n = int(line.split()[-1])
            elif line.startswith(b'property'):
                props.append(line.split()[-1].decode())
            elif line == b'end_header':
                break
        ncol = len(props)
        data = np.frombuffer(f.read(n * ncol * 4), dtype='<f4').reshape(n, ncol)
    return data, props

def mean_map_entropy(pts, radius=0.3, sample=20000, kmin=10):
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    idx = np.random.default_rng(0).choice(len(pts), min(sample, len(pts)), replace=False)
    ent = []
    for i in idx:
        nb = tree.query_ball_point(pts[i], radius)
        if len(nb) < kmin:
            continue
        cov = np.cov(pts[nb].T)
        det = np.linalg.det(cov + 1e-12 * np.eye(3))
        if det <= 0:
            continue
        ent.append(0.5 * np.log((2 * np.pi * np.e) ** 3 * det))
    return float(np.mean(ent)), len(ent)

def main():
    ply, outdir = sys.argv[1], sys.argv[2]
    traj = sys.argv[3] if len(sys.argv) > 3 else None
    import os; os.makedirs(outdir, exist_ok=True)
    data, props = read_ply(ply)
    xyz = data[:, :3]
    m = {}
    m['n_points'] = len(xyz)
    mn, mx = xyz.min(0), xyz.max(0)
    m['extent_m'] = {'x': float(mx[0]-mn[0]), 'y': float(mx[1]-mn[1]), 'z': float(mx[2]-mn[2])}

    # floor: robust low percentile (sensor-frame world; floor is the lowest dense surface)
    z = xyz[:, 2]
    floor_z = float(np.percentile(z, 1.0))
    ceil_z = float(np.percentile(z, 99.0))
    # refine floor as the densest horizontal plane within 0.4m of the low percentile
    low = xyz[z < floor_z + 0.4]
    if len(low) > 50:
        hist, edges = np.histogram(low[:, 2], bins=80)
        floor_z = float(edges[hist.argmax()])
    floor_band = xyz[np.abs(z - floor_z) < 0.06]
    m['floor_z'] = floor_z
    m['ceil_z'] = ceil_z
    m['floor_planarity_std_m'] = float(floor_band[:, 2].std()) if len(floor_band) > 50 else None

    # wall band (above floor, below ceiling) for top-down consistency check
    wall = xyz[(z > floor_z + 0.20) & (z < min(ceil_z - 0.3, floor_z + 2.5))]
    m['wall_band_points'] = len(wall)
    # f110-relevant slice near the sensor/trajectory height (NOT map median)
    sensor_z = None
    if traj:
        sensor_z = float(np.median(np.loadtxt(traj)[:, 3]))
    if sensor_z is None:
        sensor_z = floor_z + 0.4
    slice_pts = xyz[np.abs(z - sensor_z) < 0.25]
    m['slice_z'] = sensor_z

    # wall-thickness proxy: perpendicular spread of local 2D neighborhoods in the slice
    def wall_thickness(p2d, radius=0.25, sample=8000, kmin=8):
        from scipy.spatial import cKDTree
        if len(p2d) < kmin + 1:
            return None
        tree = cKDTree(p2d)
        idx = np.random.default_rng(1).choice(len(p2d), min(sample, len(p2d)), replace=False)
        spreads = []
        for i in idx:
            nb = tree.query_ball_point(p2d[i], radius)
            if len(nb) < kmin:
                continue
            c = np.cov(p2d[nb].T)
            ev = np.linalg.eigvalsh(c)
            # only count locally-linear (wall-like) neighborhoods: elongated, not blobby
            if ev[1] <= 1e-9 or ev[0] / ev[1] > 0.25:
                continue
            spreads.append(np.sqrt(max(ev[0], 0)))  # smaller eigenvalue = across-wall
        if not spreads:
            return None
        s = np.array(spreads)
        return {'p25': float(2*np.percentile(s,25)), 'p50': float(2*np.median(s)),
                'p75': float(2*np.percentile(s,75)), 'n': len(s)}
    m['wall_thickness_m'] = wall_thickness(slice_pts[:, :2])

    # Definitive wall sharpness: RANSAC line fit on the longest straight walls,
    # report perpendicular residual RMS of inliers (true wall thickness).
    def ransac_wall_rms(p2d, thresh=0.10, min_inliers=300, n_walls=6, iters=400):
        rng = np.random.default_rng(2)
        pts = p2d.copy()
        out = []
        for _ in range(n_walls):
            if len(pts) < min_inliers:
                break
            best_in, best_nd = None, None
            for _ in range(iters):
                i, j = rng.choice(len(pts), 2, replace=False)
                d = pts[j] - pts[i]
                L = np.hypot(*d)
                if L < 1.0:
                    continue
                nrm = np.array([-d[1], d[0]]) / L
                dist = np.abs((pts - pts[i]) @ nrm)
                inl = dist < thresh
                if best_in is None or inl.sum() > best_in.sum():
                    best_in, best_nd = inl, nrm
            if best_in is None or best_in.sum() < min_inliers:
                break
            wp = pts[best_in]
            # refit perpendicular via PCA, measure across-line RMS
            cc = wp - wp.mean(0)
            ev, evec = np.linalg.eigh(np.cov(cc.T))
            perp = cc @ evec[:, 0]
            length = (cc @ evec[:, 1]).ptp()
            out.append({'rms_m': float(perp.std()), 'len_m': float(length), 'n': int(best_in.sum())})
            pts = pts[~best_in]
        return out
    walls = ransac_wall_rms(slice_pts[:, :2])
    m['straight_walls'] = walls
    if walls:
        m['wall_rms_median_m'] = float(np.median([w['rms_m'] for w in walls]))

    # Mean Map Entropy (sharpness; lower=crisper) on wall+floor structure
    mme, used = mean_map_entropy(xyz, radius=0.3)
    m['mean_map_entropy'] = mme
    m['mme_samples_used'] = used

    # trajectory z drift
    if traj:
        t = np.loadtxt(traj)
        m['traj_z_range_m'] = float(t[:, 3].max() - t[:, 3].min())
        m['traj_len_xy_m'] = float(np.abs(np.diff(t[:, 1:3], axis=0)).sum())
        # start/end gap (single-pass: revisit of start area not guaranteed)
        m['traj_start_end_dist_m'] = float(np.linalg.norm(t[0, 1:4] - t[-1, 1:4]))

    with open(f'{outdir}/metrics.json', 'w') as f:
        json.dump(m, f, indent=2)

    # ---- renders ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    res = 0.02
    def topdown(pts, fname, title):
        if len(pts) < 10:
            return
        gx = ((pts[:, 0] - mn[0]) / res).astype(int)
        gy = ((pts[:, 1] - mn[1]) / res).astype(int)
        W = int((mx[0]-mn[0])/res)+1; H = int((mx[1]-mn[1])/res)+1
        grid = np.zeros((H, W))
        np.add.at(grid, (gy, gx), 1)
        plt.figure(figsize=(min(22, W/50+2), min(22, H/50+2)))
        plt.imshow(np.clip(grid, 0, 5), origin='lower', cmap='inferno', interpolation='nearest')
        plt.title(title)
        plt.tight_layout(); plt.savefig(f'{outdir}/{fname}', dpi=140); plt.close()

    topdown(wall, 'topdown.png', f'top-down wall band  MME={mme:.3f}  floor_std={m["floor_planarity_std_m"]}')
    topdown(slice_pts, 'topdown_slice.png', f'f110 slice @z={sensor_z:.2f}±0.25  (ghosting=double walls)')

    # side view (x-z): floor flatness / vertical drift
    plt.figure(figsize=(14, 5))
    plt.scatter(xyz[::5, 0], xyz[::5, 2], s=0.2, c=xyz[::5, 1], cmap='viridis')
    plt.gca().set_aspect('equal'); plt.xlabel('x'); plt.ylabel('z')
    plt.title('side view x-z (color=y)  z_extent=%.2fm' % m['extent_m']['z'])
    plt.tight_layout(); plt.savefig(f'{outdir}/side_xz.png', dpi=130); plt.close()

    print(json.dumps(m, indent=2))

if __name__ == '__main__':
    main()
