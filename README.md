# CATCH Analysis Tools

Data analysis tools for CATCH, an astronomical survey search tool

## Calibration previews and scientific coordinates

Astrometry and photometry responses publish `preview_url`, `preview_s3_key`,
`preview_orientation`, and `preview_renderer_version` alongside their FITS
artifact. Each PNG is derived from that exact artifact's full WCS, on a centered
ICRS TAN grid with north up/east left, and contains AVM for its rendered pixels.
Rendering preserves the scientific FITS bytes. Nearest-neighbor resampling is
for presentation; calculations continue to use native FITS data and zero-based
array coordinates. Native pixel diagnostics use a lower origin and a WCS-derived
N/E compass, so their pixel axes retain their scientific meaning.

Install the updated Pillow, pyavm, and reproject dependencies before deployment.
The result cache is versioned `v3-north-up`; older calibration results must be
recomputed to obtain paired previews. Cached preview URLs are refreshed from
their S3 keys, just like scientific artifact URLs. Deploy CAT before the updated
GUI; see the GUI README for the cross-service rollout sequence.

Focused checks:

```bash
python -m pytest --no-cov catch_analysis_tools/tests/north_up_preview_test.py catch_analysis_tools/tests/calibration_routes_test.py catch_analysis_tools/tests/astrometry_readiness_test.py
```

## Development and testing

Get required packages useful for development and testing:

```
pip install .[dev,tests]
```

### Code formatting and linting rules

There is a pre-commit hook to enforce code format, linting, and check for a few other things (see `.pre-commmit-config.yaml`).  It will automatically run when you `git commit` your code.  Alternatively, the format and linting checks may be run separately on any changed files:

```
ruff format
```

```
ruff check
```


### Running the webapp (without Docker)

With the package and its dependencies installed as above, the app may be started with the command:

```
python3 -m catch_analysis_tools.app.app
```

Then, to view the Swagger documentation, open up http://localhost:8000/ui (see command output for exact port).

### Running the test suite

```
tox -e py312-test
```

Running the tests with remote tests enabled, here for python 3.12:

```
tox -e py312-test -- --remote-data
```

Replacing 312 with something appropriate for your system. A list of all testing environments can be viewed with

```
tox -l
```

## Hosting

The CAT may be run as a containerized service, hosted locally or by AWS Fargate.

### Local

Running locally will install the currently checked out version of the CAT.

1. Copy env-template to .env and edit.
   a. Set TF_VAR_CAT_DEPLOYMENT="local"
2. bash \_docker

### DWD's changes:

- Added terraform configuration. For now, we are depending on my local machine for state-files.
- Moved docker stuff to just two files in the root dir: Dockerfile and docker-compose.yml. We'll worry about different deployments later.
- To run the app locally, use `docker-compose up`
- To deploy to AWS:
  - Don't do this for now -- rely on DWD to do it -- this is just FYI
  - Update the tag in .env
  - Rebuild with e.g. `docker-compose build`
  - Push to ECR with `_push_ecr_image`
  - Update tf state with `./_tf apply`
- I also made some changes adding a /hello route, and wiring up the flask code to get it to work
- Added a simple script to ping the endpoint created by tf, `./_ping_endpoint`
- To make sure that the /astrometry endpoint does not run before the Astrometry.net index files are downloaded, I added a bunch of logic in `catch_analysis_tools/app/astrometry_readiness`. See README therein for details. It also involves a new route `/health` that lets you know if the files are downloaded.


## Astrometry Configuration

The astrometric calibration pipeline depends on **astrometry.net** index files and a corresponding configuration file. These are required for WCS solving.

### 1. Install system dependencies

```
sudo apt install astrometry.net netpbm
```

### 2. Restore the astrometry index files locally

Put the restored `index-*.fits` files in:

```text
local-data/astrometry-indexes/
```

The directory contents are ignored by both Git and Docker's build context.
Create the inventory and checksums after the restore has finished:

```bash
./_upload_astrometry_indexes --manifest-only
```

`docker compose up` bind-mounts this directory directly at
`/root/.astrometry/data`, so local runs use the restored files without copying
or downloading them again.

### 3. Publish the canonical index set to S3

Terraform creates a private S3 bucket and grants the ECS task read-only access.
After applying Terraform, upload the local index set with:

```bash
./_upload_astrometry_indexes
```

The script skips files whose size and SHA-256 metadata already match and
uploads the manifest last. In Fargate, the startup readiness process downloads
that manifest and seeds the shared EFS directory from the private bucket.

### 4. Create astrometry configuration file

```bash
mkdir -p `dirname $ASTROMETRY_CONFIG`
cat > $ASTROMETRY_CONFIG <<EOF
cpulimit 300
add_path $ASTROMETRY_INDEX_DIR
autoindex
EOF
```
