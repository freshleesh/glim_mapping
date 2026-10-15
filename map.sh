#!/usr/bin/env bash
# bag -> GLIM map (.pcd + 2D track .pgm/.yaml).  Usage: ./map.sh <config.yaml> [--force]
set -e
exec python3 "$(dirname "$(readlink -f "$0")")/glim_map_pipeline.py" "$@"
