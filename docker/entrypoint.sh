#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

export ASTROMETRY_INDEX_DIR="${ASTROMETRY_INDEX_DIR:-/root/.astrometry/data}"
export ASTROMETRY_CONFIG="${ASTROMETRY_CONFIG:-/app/docker/astrometry.cfg}"

mkdir -p "${ASTROMETRY_INDEX_DIR}"

exec python3 -m catch_analysis_tools.app.app
