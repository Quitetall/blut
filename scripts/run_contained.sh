#!/usr/bin/env bash
# run_contained.sh — run a long/training command in a MEMORY-CAPPED,
# session-detached transient systemd --user service. Two guarantees:
#
#   1. OOM CONTAINMENT. A RAM blowup (e.g. a DataLoader fork-worker
#      copy-on-write of a large parent process, ×N workers) is
#      cgroup-OOM-killed in ISOLATION. It cannot trigger the GLOBAL kernel
#      OOM that would sweep the whole user session (monitors, shells, the
#      training job, the editor) at once.
#   2. SESSION-SIGTERM SURVIVAL. The work runs in its own systemd slice,
#      immune to the session-teardown SIGTERM that kills nohup'd
#      tool-pgroup children (nohup blocks SIGHUP, not SIGTERM).
#
# This script is the contained-launch backend for blut's `LocalSystemd`
# launcher; the broker normally sets the authoritative MEMMAX/MEMHIGH/SWAPMAX
# from a stage's typed footprint, but the env knobs below are honoured for a
# manual `bash run_contained.sh ...` invocation too.
#
# Env knobs:
#   UNIT     transient unit name            (default blut-<epoch>)
#   MEMMAX   hard cap; cgroup-OOM if hit    (default 44G)
#   MEMHIGH  soft cap; reclaim pressure     (default 40G)
#   SWAPMAX  swap ceiling; small=fail-fast  (default 2G — kept small so a
#            residual overshoot OOM-kills the unit instead of thrashing host
#            swap; the working set must fit MEMMAX, not swap.)
#
# Usage:
#   run_contained.sh python -u train.py ...
#   UNIT=distill MEMMAX=30G run_contained.sh python -u train.py ...
# Inspect:  journalctl --user -u <UNIT> -f   |   systemctl --user status <UNIT>
# Stop:     systemctl --user stop <UNIT>
set -euo pipefail
UNIT="${UNIT:-blut-$(date +%s)}"
# --user services run in the user-manager's MINIMAL env (no ~/.cargo/bin, no
# caller HOME). Propagate the caller's PATH + HOME so binaries on the shell
# PATH and per-user caches resolve.
systemd-run --user --unit="$UNIT" --collect \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  -p MemoryAccounting=yes \
  -p MemoryMax="${MEMMAX:-44G}" \
  -p MemoryHigh="${MEMHIGH:-40G}" \
  -p MemorySwapMax="${SWAPMAX:-2G}" \
  -- "$@"
echo "[run_contained] unit=$UNIT  MemoryMax=${MEMMAX:-44G} MemoryHigh=${MEMHIGH:-40G} SwapMax=${SWAPMAX:-2G}"
echo "[run_contained] watch: journalctl --user -u $UNIT -f   status: systemctl --user status $UNIT"
