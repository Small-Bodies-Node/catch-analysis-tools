rovide the newly added section in code formatr# CATCH Analysis Tools

Data analysis tools for CATCH, an astronomical survey search tool

## Testing

Get required packages useful for testing:

```
pip install .[tests]
```

Running the tests:

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
- Moved docker stuff to just two files in the root dir: Dockerfile and docker-compose.yml. We'll worry about different deployments later
- To run the app locally, use `docker-compose up`
- To deploy to AWS:
  - Don't do this for now -- rely on DWD to do it -- this is just FYI
  - Update the tag in .env
  - Rebuild with e.g. `docker-compose build`
  - Push to ECR with `_push_ecr_image`
  - Update tf state with `./_tf apply`
- I also made some changes adding a /hello route, and wiring up the flask code to get it to work
- Added a simple script to ping the endpoint created by tf, `./_ping_endpoint`


## Astrometry Configuration 

The astrometric calibration pipeline depends on **astrometry.net** index files and a corresponding configuration file. These are required for WCS solving.

### 1. Install system dependencies
```
sudo apt install astrometry.net netpbm
```

### 2. Download astrometry index files
```
mkdir -p ~/.astrometry/data

for i in 00 01 02 03 04 05 06 07; do
  wget -P ~/.astrometry/data \
    https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/index-5204-$i.fits
done

```
Note: These files are required and may take time to download (~GB total).

### 3. Create astrometry configuration file
```
mkdir -p ~/.astrometry
: > ~/.astrometry/config

for f in ~/.astrometry/data/index-*.fits; do
  echo "index $f" >> ~/.astrometry/config
done
```
### 4. Set required environment variable
```
export ASTROMETRY_CONFIG=$HOME/.astrometry/config
```
