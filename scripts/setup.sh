#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3.11}" -c 'import sys; sys.exit("Python 3.11 is required") if sys.version_info[:2] != (3, 11) else None'
"${PYTHON:-python3.11}" -m venv .venv
case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) wheel=macosx_11_0_arm64 ;;
    Linux/aarch64) wheel=manylinux_2_28_aarch64 ;;
    Linux/x86_64) wheel=manylinux_2_28_x86_64 ;;
    *) printf '%s\n' 'This platform has no pinned AISimulate wheel.' >&2; exit 1 ;;
esac
.venv/bin/python -m pip install -r requirements.txt
# The study uses the wheel's CPU operator engine. JAX, tuning, and server
# extras from AISimulate are outside the simulator path.
.venv/bin/python -m pip install --no-deps -r "scripts/aisimulate-$wheel.txt"
