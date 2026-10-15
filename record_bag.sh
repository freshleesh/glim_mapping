#!/usr/bin/env bash
# Record a mapping bag (Livox Mid-360 LiDAR + IMU only — all GLIM needs).
# Usage: ./record_bag.sh <name> [extra ros2 bag record args/topics]
#        ./record_bag.sh <name> -a        # everything, like `ros2 bag record -a`
# Run on the machine that sees the livox topics (car, or a PC on the same DDS domain).
# Drive slowly, close at least one loop, end where you started.
set -e
NAME="${1:?usage: record_bag.sh <name> [extra args]}"; shift
TOPICS=(/livox/lidar /livox/imu)
for a in "$@"; do [ "$a" = "-a" ] && TOPICS=(); done
ros2 bag record -s mcap -o "$NAME" "${TOPICS[@]}" "$@"
