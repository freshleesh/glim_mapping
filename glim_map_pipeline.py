#!/usr/bin/env python3
"""GLIM offline mapping from a bag -> a named map, driven by one small YAML config.

Give it a map name + bag, run once, and you get exactly two products in
<out_root>/<map_name>/ (everything else stays in a hidden .work/ subdir):
  <map_name>.pcd          full 3D cloud  (kiss_icp_localization map_pcd_3d)
  <map_name>.pgm/.yaml     2D grid  (map_server /map + kiss detection track mask)

Steps (all GLIM in Docker): map the bag -> export the merged cloud -> write the
.pcd -> find the ground height (GLIM already gravity-levels the map, so this is
just the floor z + tiny residual tilt) -> keep the [z_min, z_max] wall band ->
rasterize + flood-fill from the trajectory -> the track mask.

Usage:  python3 glim_map_pipeline.py <config.yaml> [--force]
Config keys:
  map_name:       hall          # output name (folder + file basenames)
  bag_path:       /abs/rosbag2_dir  (or a .mcap file)
  # optional (defaults shown):
  out_root:       <repo>/maps   # <out_root>/<map_name>/ is created (relative = to the config file)
  z_min:          0.05          # wall band, m above the floor
  z_max:          0.30
  png_resolution: 0.05          # m / pixel for the track grid
  use_gpu:        true          # true -> config/gpu + cuda image, false -> config/cpu
  glim_config_dir: config/gpu_lowmem   # override the GLIM config dir (relative = to the repo)
  occ_min_pts:    3             # min band points per cell to call it a wall
  dense_z:        0.4           # height split: <=dense_z (track) dense, above sparse
  voxel_dense:    0.05          # voxel (m) for z<=dense_z (0 = no downsample)
  voxel_sparse:   0.25          # voxel (m) for z>dense_z
"""
import sys, os, glob, argparse, subprocess
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{REPO}/tools')
from eval_map import read_ply  # reuse the binary-PLY reader

# --- locations, all relative to this repo ---
CFG_CPU      = f'{REPO}/config/cpu'
CFG_GPU      = f'{REPO}/config/gpu'
IMG_MAP_CPU  = 'koide3/glim_ros2:jazzy'
IMG_MAP_GPU  = 'koide3/glim_ros2:jazzy_cuda13.1'
IMG_EXPORT   = 'glim_tools:jazzy'   # built from tools/export_ply/Dockerfile (has glim_export_ply)


# ------------------------------------------------------------------ docker helper
def _docker_direct():
    """True if this shell can already talk to docker (user in docker group, rootless, ...)."""
    return subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


DOCKER_DIRECT = None


def sh(cmd, log=None):
    """Run a docker `cmd`, streaming output. Falls back to `sg docker -c` when the
    docker group was just added and this shell hasn't picked it up yet."""
    global DOCKER_DIRECT
    if DOCKER_DIRECT is None:
        DOCKER_DIRECT = _docker_direct()
    print(f'\n$ {cmd}\n', flush=True)
    argv = ['bash', '-c', cmd] if DOCKER_DIRECT else ['sg', 'docker', '-c', cmd]
    p = subprocess.Popen(argv,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
        if log:
            log.write(line); log.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')


def chown_back(out_dir, image):
    """Docker writes as root; hand the files back to the host user."""
    sh(f'docker run --rm -v {out_dir}:/out {image} chown -R {os.getuid()}:{os.getgid()} /out')


def ensure_export_image():
    """Build the PLY-exporter image on first use (new machine)."""
    q = f'docker image inspect {IMG_EXPORT} >/dev/null 2>&1 || docker build -t {IMG_EXPORT} {REPO}/tools/export_ply'
    sh(q)


# ------------------------------------------------------------------ PCD writer
def write_pcd(path, xyz, intensity=None):
    """Write a binary PCD (x y z [intensity], float32)."""
    n = len(xyz)
    if intensity is not None:
        fields, cols = 'x y z intensity', np.column_stack([xyz, intensity]).astype('<f4')
        sizes, types, counts = '4 4 4 4', 'F F F F', '1 1 1 1'
    else:
        fields, cols = 'x y z', np.ascontiguousarray(xyz, dtype='<f4')
        sizes, types, counts = '4 4 4', 'F F F', '1 1 1'
    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        f'FIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n'
        f'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {n}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(cols).tobytes())
    print(f'  wrote {path}  ({n} pts)')


# ------------------------------------------------------------------ voxel downsample
def voxel_downsample(xyz, intensity, voxel):
    """Keep one centroid point per `voxel`-metre cube. voxel<=0 -> passthrough."""
    if voxel is None or voxel <= 0.0 or len(xyz) == 0:
        return xyz, intensity
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(cnt), 3), np.float64)
    np.add.at(sums, inv, xyz)
    ds_xyz = (sums / cnt[:, None]).astype(np.float32)
    ds_int = None
    if intensity is not None:
        isum = np.zeros(len(cnt), np.float64)
        np.add.at(isum, inv, intensity)
        ds_int = (isum / cnt).astype(np.float32)
    return ds_xyz, ds_int


def stratified_downsample(xyz, intensity, height, dense_z, voxel_dense, voxel_sparse):
    """Height-stratified density: points at/below `dense_z` (above the floor) are the
    track region -> keep dense (voxel_dense); points above are walls/ceiling -> sparse
    (voxel_sparse). Returns merged (xyz, intensity)."""
    low = height <= dense_z
    lo_i = intensity[low] if intensity is not None else None
    hi_i = intensity[~low] if intensity is not None else None
    lo_xyz, lo_int = voxel_downsample(xyz[low], lo_i, voxel_dense)
    hi_xyz, hi_int = voxel_downsample(xyz[~low], hi_i, voxel_sparse)
    m_xyz = np.vstack([lo_xyz, hi_xyz]) if len(hi_xyz) else lo_xyz
    if intensity is not None:
        m_int = np.concatenate([lo_int, hi_int]) if len(hi_xyz) else lo_int
    else:
        m_int = None
    print(f'  density split @ height={dense_z}m: below={len(lo_xyz)} (voxel {voxel_dense}m) '
          f'| above={len(hi_xyz)} (voxel {voxel_sparse}m)')
    return m_xyz, m_int


# ------------------------------------------------------------------ ground plane
def fit_ground_plane(xyz, dist_thresh=0.05, iters=800, vertical_min=0.85, seed=0):
    """RANSAC-fit the ground plane. Returns (normal[3] pointing up, d) for n.x + d = 0.
    height above ground of a point p  =  n.p + d  (signed; ground ~ 0)."""
    rng = np.random.default_rng(seed)
    z = xyz[:, 2]
    floor_z = np.percentile(z, 1.0)
    cand = xyz[z < floor_z + 0.6]
    if len(cand) < 100:
        cand = xyz
    if len(cand) > 80000:                       # subsample for fitting speed
        cand = cand[rng.choice(len(cand), 80000, replace=False)]
    best_cnt, best = -1, None
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) < vertical_min:             # reject walls / tilted planes
            continue
        d = -n @ p[0]
        inl = np.abs(cand @ n + d) < dist_thresh
        if inl.sum() > best_cnt:
            best_cnt, best = inl.sum(), (n, d, inl)
    if best is None:                             # fallback: flat plane at floor_z
        return np.array([0.0, 0.0, 1.0]), -floor_z
    n, d, inl = best
    pin = cand[inl]                              # PCA refit on inliers
    c = pin.mean(0)
    _, _, vt = np.linalg.svd(pin - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, float(-n @ c)


# ------------------------------------------------------------------ track mask
def leveling_rotation(pts):
    """Rotation R that levels the best-fit plane of `pts` (normal -> +z), plus the
    tilt angle removed. Reference = the CAR TRAJECTORY: the car drives on the flat
    floor, so its path is a cleaner horizontal than GLIM's residual gravity-level or
    a ground RANSAC (walls/clutter bias). Fixes the ~1 deg map tilt that pushes the
    floor into the detection z-band."""
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    abc = np.linalg.lstsq(A, pts[:, 2], rcond=None)[0]
    inl = np.abs(pts[:, 2] - A @ abc) < 0.1
    a, b, _ = np.linalg.lstsq(A[inl], pts[inl, 2], rcond=None)[0]
    n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0]); ax = np.cross(n, z); sN = np.linalg.norm(ax)
    if sN < 1e-9:
        return np.eye(3), 0.0
    ax /= sN; ang = float(np.arccos(np.clip(n @ z, -1.0, 1.0)))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return R, float(np.degrees(ang))


def build_track_mask(obst_xy, traj_xy, out_prefix, res, min_pts=3):
    """2D drivable-region mask: occupied = wall-band points, free = flood-fill from
    the trajectory sealed by those walls. Writes only <out_prefix>.pgm/.yaml
    (+ .png preview) as a ROS map_server grid (track=255 white, else 0), which
    kiss_icp_localization loads to reject detections off the track."""
    from scipy.ndimage import binary_dilation, binary_closing
    from collections import deque
    allxy = np.vstack([obst_xy, traj_xy])
    mn = allxy.min(0) - 0.5
    mx = allxy.max(0) + 0.5
    W = int((mx[0] - mn[0]) / res) + 1
    H = int((mx[1] - mn[1]) / res) + 1

    def to_cell(p):
        return (((p[:, 0] - mn[0]) / res).astype(int),
                ((p[:, 1] - mn[1]) / res).astype(int))

    count = np.zeros((H, W), np.int32)
    gx, gy = to_cell(obst_xy)
    ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    np.add.at(count, (gy[ok], gx[ok]), 1)
    occ = count >= min_pts
    barrier = binary_closing(binary_dilation(occ, iterations=2), structure=np.ones((3, 3)), iterations=2)

    free = np.zeros((H, W), bool)
    seen = np.zeros((H, W), bool)
    seeds = deque()
    tgx, tgy = to_cell(traj_xy)
    for x, y in zip(tgx, tgy):
        if 0 <= x < W and 0 <= y < H and not occ[y, x]:
            seeds.append((y, x))
    while seeds:
        y, x = seeds.popleft()
        if seen[y, x] or barrier[y, x]:
            continue
        seen[y, x] = True
        free[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and not barrier[ny, nx]:
                seeds.append((ny, nx))

    img = np.flipud(np.where(free, 255, 0).astype(np.uint8))  # PGM top-left -> ROS y-up
    with open(out_prefix + '.pgm', 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (W, H))
        f.write(img.tobytes())
    name = os.path.basename(out_prefix)
    with open(out_prefix + '.yaml', 'w') as f:
        f.write(f'image: {name}.pgm\nresolution: {res}\n'
                f'origin: [{mn[0]:.4f}, {mn[1]:.4f}, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(out_prefix + '.png', img, cmap='gray', vmin=0, vmax=255)
    print(f'  track {W}x{H} @ {res}m  walls={int(occ.sum())} free={int(free.sum())}  -> {out_prefix}.pgm')


# ------------------------------------------------------------------ main
def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('config', help='map YAML config (see map_config.example.yaml)')
    ap.add_argument('--force', action='store_true', help='redo every stage even if outputs exist')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))

    def path_of(v, base):
        v = os.path.expanduser(str(v))
        return os.path.abspath(v if os.path.isabs(v) else os.path.join(base, v))

    map_name = str(cfg['map_name']).strip()
    bag_path = path_of(cfg['bag_path'], cfg_dir)
    out_root = path_of(cfg.get('out_root', f'{REPO}/maps'), cfg_dir)
    z_min    = float(cfg.get('z_min', 0.05))
    z_max    = float(cfg.get('z_max', 0.30))
    res      = float(cfg.get('png_resolution', 0.05))
    use_gpu  = bool(cfg.get('use_gpu', True))
    glim_cfg = path_of(cfg.get('glim_config_dir', CFG_GPU if use_gpu else CFG_CPU), REPO)
    img_map  = IMG_MAP_GPU if use_gpu else IMG_MAP_CPU
    occ_min  = int(cfg.get('occ_min_pts', 3))
    # height-stratified pcd density: dense below dense_z (track), sparse above (walls)
    dense_z      = float(cfg.get('dense_z', 0.4))
    voxel_dense  = float(cfg.get('voxel_dense', 0.05))
    voxel_sparse = float(cfg.get('voxel_sparse', 0.25))
    assert z_min < z_max, 'z_min must be < z_max'

    # <out_root>/<map_name>/ holds the 2 products; .work/ holds the GLIM intermediates.
    map_dir  = f'{out_root}/{map_name}'
    work_dir = f'{map_dir}/.work'
    dump_dir = f'{work_dir}/dump'
    merged   = f'{work_dir}/merged.ply'
    os.makedirs(dump_dir, exist_ok=True)
    log = open(f'{work_dir}/pipeline.log', 'a')
    pcd_out   = f'{map_dir}/{map_name}.pcd'
    track_out = f'{map_dir}/{map_name}'   # the 2D grid: serves map_server /map AND kiss track mask
    print(f'=== GLIM map "{map_name}" ===\n  bag={bag_path}\n  out={map_dir}\n'
          f'  glim_config={glim_cfg} ({"GPU" if use_gpu else "CPU"})\n'
          f'  wall band z=[{z_min}, {z_max}] m   res={res} m/px')

    # locate bag: prefer an .mcap file, else a .db3 dir, else the path as given
    if bag_path.endswith('.mcap'):
        bag_dir, bag_arg = os.path.dirname(bag_path), '/bag/' + os.path.basename(bag_path)
    else:
        bag_dir = bag_path
        mcap = sorted(glob.glob(f'{bag_dir}/*.mcap'))
        db3  = sorted(glob.glob(f'{bag_dir}/*.db3'))
        if mcap:
            bag_arg = '/bag/' + os.path.basename(mcap[0])
        elif db3:
            bag_arg = '/bag'              # sqlite3 bag: glim_rosbag takes the directory
        else:
            raise SystemExit(f'no .mcap/.db3 under {bag_dir}')

    # ---- stage 1: mapping (Docker GLIM) ----
    have_dump = os.path.isdir(dump_dir) and os.listdir(dump_dir)
    if args.force or not have_dump:
        os.makedirs(dump_dir, exist_ok=True)
        print('\n[1/3] mapping bag with saved GLIM settings ...')
        gpus = '--gpus all ' if use_gpu else ''
        sh(f'docker run --rm {gpus}-v {bag_dir}:/bag:ro -v {glim_cfg}:/cfg:ro -v {work_dir}:/out '
           f'{img_map} ros2 run glim_ros glim_rosbag {bag_arg} '
           f'--ros-args -p config_path:=/cfg -p dump_path:=/out/dump -p auto_quit:=true',
           log=log)
        chown_back(work_dir, img_map)
    else:
        print('\n[1/3] dump exists -> skip mapping (use --force to redo)')

    # ---- stage 2: export merged global cloud (dump -> PLY) ----
    if args.force or not os.path.isfile(merged):
        print('\n[2/3] exporting merged cloud ...')
        ensure_export_image()
        sh(f"docker run --rm -v {work_dir}:/out -v {glim_cfg}:/cfg:ro {IMG_EXPORT} bash -lc '"
           '. /opt/ros/jazzy/setup.sh; . /root/ros2_ws/install/setup.sh; '
           "glim_export_ply /out/dump /out/merged.ply /cfg'", log=log)
        chown_back(work_dir, IMG_EXPORT)
    else:
        print('\n[2/3] merged.ply exists -> skip export')

    data, props = read_ply(merged)
    xyz = data[:, :3]
    intensity = data[:, props.index('intensity')] if 'intensity' in props else None

    # --- trajectory-based leveling (removes GLIM's ~1 deg residual map tilt) ---
    _tp = f'{dump_dir}/traj_lidar.txt'
    traj_full = np.loadtxt(_tp)[:, 1:4] if os.path.isfile(_tp) else None
    traj_xy = None
    if cfg.get('level_to_trajectory', True) and traj_full is not None and len(traj_full) > 50:
        R, ang = leveling_rotation(traj_full)
        ctr = traj_full.mean(0)
        xyz = ((xyz - ctr) @ R.T + ctr).astype(np.float32)
        traj_full = (traj_full - ctr) @ R.T + ctr
        print(f'  trajectory-leveled: removed {ang:.2f} deg map tilt (car-path reference)')
    if traj_full is not None:
        traj_xy = traj_full[:, :2]

    # ground plane (shared by the height-stratified pcd density AND the wall band).
    # GLIM gravity-levels the map, so this is just the floor z + residual tilt (RANSAC,
    # near-vertical). height above ground of a point = n.p + d.
    n, d = fit_ground_plane(xyz)
    height = xyz @ n + d
    tilt_deg = float(np.degrees(np.arccos(min(1.0, abs(n[2])))))
    print(f'  ground: tilt={tilt_deg:.2f}deg (GLIM-leveled)')

    # product 1: 3D map, height-stratified density — dense on the track (<= dense_z),
    # sparse for the walls/ceiling above it.
    m_xyz, m_int = stratified_downsample(xyz, intensity, height,
                                         dense_z, voxel_dense, voxel_sparse)
    write_pcd(pcd_out, m_xyz, m_int)

    # ---- stage 3: wall band -> 2D track mask (uses the FULL cloud, not downsampled) ----
    print('\n[3/3] wall band + track mask ...')
    keep = (height >= z_min) & (height <= z_max)
    print(f'  wall-band pts={int(keep.sum())}/{len(xyz)}')
    if traj_xy is None:
        traj_xy = xyz[keep][:, :2]
    build_track_mask(xyz[keep][:, :2], traj_xy, track_out, res, min_pts=occ_min)

    print('\n=== done ===')
    print(f'  {pcd_out}          (3D map -> kiss map_pcd_3d)')
    print(f'  {track_out}.pgm/.yaml  (2D grid -> map_server /map + kiss track mask)')
    print(f'  (GLIM intermediates cached in {work_dir})')
    log.close()


if __name__ == '__main__':
    main()
