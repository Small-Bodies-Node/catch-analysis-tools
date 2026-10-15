# Local Astrometry.net indexes

Copy the canonical `index-*.fits` files into this directory. The FITS files are
ignored by Git and excluded from the Docker build context.

Create the checksum manifest without uploading anything:

```bash
./_upload_astrometry_indexes --manifest-only
```

Run the application against these files:

```bash
docker compose up
```

After Terraform has created the private reference-data bucket, upload or resume
the S3 copy:

```bash
./_upload_astrometry_indexes
```

The upload is safe to rerun. Objects whose size and SHA-256 metadata already
match are skipped. The manifest is uploaded last, so production does not treat
an incomplete index set as ready.
