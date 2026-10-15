#!/usr/bin/env python3
"""Decimate the LiDAR topic of a rosbag to simulate high-speed driving.

Keeps every Nth /livox/lidar frame, leaves /livox/imu (and tf/odom) untouched,
and drops bulky image topics. Messages are copied as raw serialized bytes, so
the message type definitions do NOT need to be installed.

Effect: at constant true speed, dropping LiDAR frames by factor N multiplies the
inter-scan displacement by N -> equivalent to driving N x faster, while the IMU
(full rate) still bridges the larger gap consistently. Isolates "large
inter-frame motion" robustness without faking IMU dynamics.

Usage:
  python3 decimate_lidar.py <in_bag_dir> <out_bag_dir> <N> [--lidar TOPIC] [--keep-images]
"""
import argparse
import os
import sys

import rosbag2_py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", help="input bag directory (contains *.mcap + metadata.yaml)")
    ap.add_argument("out_dir", help="output bag directory (must not exist)")
    ap.add_argument("N", type=int, help="keep every Nth lidar frame (effective speedup factor)")
    ap.add_argument("--lidar", default="/livox/lidar", help="lidar topic to decimate")
    ap.add_argument("--keep-images", action="store_true", help="keep image topics (default: drop)")
    args = ap.parse_args()

    if args.N < 1:
        sys.exit("N must be >= 1")
    if os.path.exists(args.out_dir):
        sys.exit(f"ERROR: {args.out_dir} already exists -- pick a fresh path")

    drop_topics = set()
    if not args.keep_images:
        drop_topics = {"/image_raw", "/image_raw/compressed"}

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.in_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=args.out_dir, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    for tm in reader.get_all_topics_and_types():
        if tm.name in drop_topics:
            continue
        writer.create_topic(tm)

    kept = dropped = 0
    lidar_i = 0
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in drop_topics:
            continue
        if topic == args.lidar:
            keep = (lidar_i % args.N == 0)
            lidar_i += 1
            if keep:
                kept += 1
            else:
                dropped += 1
                continue
        writer.write(topic, data, t)

    del writer
    print(f"lidar '{args.lidar}': kept {kept}, dropped {dropped} (N={args.N})")
    print(f"output: {args.out_dir}")


if __name__ == "__main__":
    main()
