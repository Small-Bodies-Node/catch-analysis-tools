"""
Runtime readiness helpers for Astrometry.net index files.

This package is application/deployment plumbing, not scientific astrometry
logic. The /astrometry route needs Astrometry.net index FITS files on disk
before it can safely call solve-field. In AWS Fargate that directory is backed
by EFS and mounted at /root/.astrometry/data inside the container.

The docker-compose.yml file is for local development/debugging. When running
locally with `docker compose up`, Docker bind-mounts `${HOME}/.astrometry/data`
from the host machine to /root/.astrometry/data inside the container. Missing
index files are downloaded into that host directory, so they persist across
container rebuilds and restarts.

The helpers in this package discover the expected remote index files, download
missing files, expose cheap in-memory readiness status, and use a file lock so
multiple app instances do not download the same files at the same time.
"""
