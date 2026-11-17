"""
Runtime readiness helpers for Astrometry.net index files.

This package is application/deployment plumbing, not scientific astrometry
logic. The /calibration/astrometry route needs Astrometry.net index FITS files
on disk before it can safely call solve-field. In AWS Fargate that directory is
backed by EFS and mounted at /root/.astrometry/data inside the container.

The docker-compose.yml file is for local development/debugging. When running
locally with `docker compose up`, Docker bind-mounts the repository's
local-data/astrometry-indexes directory at /root/.astrometry/data. In Fargate,
missing files are copied from a private S3 bucket into shared EFS storage.

The helpers use a checksum manifest to discover and verify the expected files,
expose cheap in-memory readiness status, and use a file lock so multiple app
instances do not download the same files at the same time.
"""
