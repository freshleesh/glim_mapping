#!/usr/bin/env python3
"""Observation-count dynamic-object removal for GLIM maps.

A static point (wall/structure) is seen from MANY overlapping GLIM submaps; a
dynamic object (a walking person during mapping) occupies a voxel in only ONE
or two submaps because it moves out. So: count, per voxel, how many DISTINCT
submaps observed it, and drop map points whose voxel was seen by fewer than
`--min-obs` submaps.

Input is a mapped folder produced by glim_map_pipeline.py:
    maps_glim/<name>/<name>.pcd            the final (stratified) 3D map
    maps_glim/<name>/.work/dump/<NNNNNN>/  per-submap points + pose (T_world_origin)

Usage:
    remove_dynamic.py <map_dir> [--voxel 0.2] [--min-obs 2] [--dilate]
                      [--out PATH | --replace]

Notes
  * submap points_compact.bin  = N x 3 float32 (xyz, submap-local frame)
  * data.txt T_world_origin     = 4x4 submap->world pose (GLIM optimized)
  * --min-obs 2 is conservative (keeps anything two submaps agree on). Raise it
    to strip more aggressively; a far wall seen from only one spot needs a low
    threshold so lower it / use --dilate if legit static gets removed.
  * --dilate keeps a map point if ANY of its 27 neighbour voxels clears the
    threshold — absorbs the sub-voxel drift between a submap's dump pose and the
    final merged pose. Costs 27x lookups; on by default is a good idea if you
    see thin static structures getting nibbled.
"""
import sys, os, glob, argparse
import numpy as np

# packing: voxel coords -> one int64 key (bijective, monotone for searchsorted).
# OFFSET makes coords non-negative; S bounds each axis to [-2^20, 2^20) voxels
# (±200 km at 0.2 m) — far beyond any real map.
_OFF = 1 << 20
_S = 1 << 21


def _pack(vk):
    a = vk.astype(np.int64) + _OFF
    return (a[:, 0] * _S + a[:, 1]) * _S + a[:, 2]


def _read_pose(data_txt):
    """Parse the 4x4 T_world_origin matrix out of a submap data.txt."""
    lines = open(data_txt).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith('T_world_origin'):
            rows = [list(map(float, lines[i + 1 + r].split())) for r in range(4)]
            return np.array(rows, dtype=np.float64)
    raise ValueError(f'no T_world_origin in {data_txt}')


def _read_pcd(path):
    """Read a binary PCD written by glim_map_pipeline (x y z [intensity], f32)."""
    with open(path, 'rb') as f:
        raw = f.read()
    nl = raw.index(b'DATA binary\n') + len(b'DATA binary\n')
    header = raw[:nl].decode('ascii', 'replace')
    fields = next(l for l in header.splitlines() if l.startswith('FIELDS')).split()[1:]
    npts = int(next(l for l in header.splitlines() if l.startswith('POINTS')).split()[1])
    body = np.frombuffer(raw[nl:nl + npts * 4 * len(fields)], dtype='<f4').reshape(npts, len(fields))
    xyz = body[:, :3].astype(np.float32)
    inten = body[:, 3].astype(np.float32) if 'intensity' in fields else None
    return xyz, inten


def _write_pcd(path, xyz, inten):
    n = len(xyz)
    if inten is not None:
        cols = np.column_stack([xyz, inten]).astype('<f4')
        fields, sizes, types, counts = 'x y z intensity', '4 4 4 4', 'F F F F', '1 1 1 1'
    else:
        cols = np.ascontiguousarray(xyz, dtype='<f4')
        fields, sizes, types, counts = 'x y z', '4 4 4', 'F F F', '1 1 1'
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
              f'FIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n'
              f'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(cols).tobytes())


def build_submap_counts(dump_dir, voxel):
    """Per voxel, how many distinct submaps observed it. Returns sorted
    (packed_keys, counts)."""
    subdirs = sorted(d for d in glob.glob(os.path.join(dump_dir, '[0-9]' * 6))
                     if os.path.isfile(os.path.join(d, 'points_compact.bin')))
    if not subdirs:
        sys.exit(f'no submaps under {dump_dir}')
    per_submap_keys = []
    for d in subdirs:
        pts = np.fromfile(os.path.join(d, 'points_compact.bin'), dtype='<f4').reshape(-1, 3)
        T = _read_pose(os.path.join(d, 'data.txt'))
        world = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        vk = np.floor(world / voxel).astype(np.int64)
        per_submap_keys.append(np.unique(_pack(vk)))   # dedup within this submap
    allk = np.concatenate(per_submap_keys)
    keys, counts = np.unique(allk, return_counts=True)  # count = #submaps per voxel
    print(f'  {len(subdirs)} submaps -> {len(keys)} occupied voxels @ {voxel} m')
    return keys, counts.astype(np.int32)


def lookup(keys, counts, mk, dilate):
    """Vectorised max-count over a point's voxel (and 27-nbhd if dilate)."""
    def _one(q):
        idx = np.searchsorted(keys, q)
        idx = np.clip(idx, 0, len(keys) - 1)
        hit = keys[idx] == q
        return np.where(hit, counts[idx], 0)
    if not dilate:
        return _one(mk)
    best = np.zeros(len(mk), np.int32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                shift = (dx * _S + dy) * _S + dz
                best = np.maximum(best, _one(mk + shift))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_dir', help='maps_glim/<name>/ (has <name>.pcd + .work/dump)')
    ap.add_argument('--voxel', type=float, default=0.2, help='voxel size [m] (default 0.2)')
    ap.add_argument('--min-obs', type=int, default=2, help='keep voxel if seen by >= this many submaps (default 2)')
    ap.add_argument('--no-dilate', dest='dilate', action='store_false',
                    help='disable the 27-nbhd rescue (removes more, but pose drift can nibble thin static)')
    ap.set_defaults(dilate=True)
    ap.add_argument('--out', help='output pcd (default <name>_clean.pcd)')
    ap.add_argument('--replace', action='store_true', help='overwrite <name>.pcd (backup .predyn.bak)')
    a = ap.parse_args()

    mdir = os.path.abspath(a.map_dir.rstrip('/'))
    name = os.path.basename(mdir)
    pcd = os.path.join(mdir, f'{name}.pcd')
    dump = os.path.join(mdir, '.work', 'dump')
    for p in (pcd, dump):
        if not os.path.exists(p):
            sys.exit(f'missing {p}')

    print(f'[remove_dynamic] {name}  voxel={a.voxel}  min_obs={a.min_obs}  dilate={a.dilate}')
    keys, counts = build_submap_counts(dump, a.voxel)

    xyz, inten = _read_pcd(pcd)
    mk = _pack(np.floor(xyz / a.voxel).astype(np.int64))
    obs = lookup(keys, counts, mk, a.dilate)
    keep = obs >= a.min_obs
    n0, n1 = len(xyz), int(keep.sum())
    print(f'  map {n0} pts -> keep {n1} ({100.0*n1/n0:.1f}%), removed {n0-n1} '
          f'(obs<{a.min_obs}); obs hist: '
          + ' '.join(f'{k}:{int((obs==k).sum())}' for k in range(0, 6)) + ' ...')

    out = a.out or os.path.join(mdir, f'{name}_clean.pcd')
    if a.replace:
        os.replace(pcd, pcd + '.predyn.bak')
        out = pcd
        print(f'  backup -> {pcd}.predyn.bak')
    _write_pcd(out, xyz[keep], inten[keep] if inten is not None else None)
    print(f'  wrote {out}  ({n1} pts)')


if __name__ == '__main__':
    main()
