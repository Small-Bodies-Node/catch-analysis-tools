# Astrometry Readiness

This package is runtime plumbing for `/astrometry`. It exists so the app does
not call Astrometry.net's `solve-field` until the required index FITS files are
present on disk.

This is not scientific astrometry code. The scientific route only taps into this plumbing with two simple questions answered by the following hooks:

```python
is_astrometry_ready()
get_astrometry_readiness_status()
```

Everything else in this package is about making those answers reliable while
the app is starting, while files are downloading, and while more than one app
instance may be running.

## Storage

Inside the container, the index files always live here:

```text
/root/.astrometry/data
```

That path is intentionally the same in every environment.

In AWS Fargate, Terraform configures ECS to mount EFS at that container path.
All Fargate tasks for this service see the same EFS-backed directory.

In local development, `docker compose up` uses this bind mount:

```yaml
${HOME}/.astrometry/data:/root/.astrometry/data
```

So local downloads are written to the host machine at:

```text
${HOME}/.astrometry/data
```

Those files persist across local container rebuilds and restarts.

## Startup Flow

The Docker entrypoint sets defaults for the index directory and Astrometry.net
config:

```bash
ASTROMETRY_INDEX_DIR=/root/.astrometry/data
ASTROMETRY_CONFIG=/app/docker/astrometry.cfg
```

Then it starts the app with:

```bash
python3 -m catch_analysis_tools.app.app
```

When `app.py` runs as `__main__`, it calls:

```python
start_astrometry_background_check()
```

That starts one daemon thread in the current Python process. This thread runs
the index-file check/download work in the background, so the HTTP app can start
immediately instead of blocking until hundreds of index files are checked or
downloaded.

While the background thread is still working:

- `/hello` can respond normally.
- `/health` reports the current readiness status.
- `/astrometry` returns `503 not_ready` instead of trying `solve-field`.

After the background thread confirms all expected index files exist,
`/astrometry` is allowed to run the scientific pipeline.

## What State Lives In RAM

Each app process has its own in-memory status object in `state.py`.

That status includes fields such as:

```text
state
ready
message
files_present
expected_files
error
updated_at
```

This RAM state is only local to one Python process. It is not shared across
Fargate tasks, Docker containers, or worker processes.

That means one process cannot directly update another process's RAM. If two
Fargate tasks are running, each task has its own background thread and its own
readiness status.

Cross-process coordination happens through the shared filesystem, not through
RAM.

## The In-Memory Thread Lock

`state.py` also owns a `threading.RLock`.

That lock only protects the current Python process. It prevents the background
thread and HTTP routes in the same process from reading/writing the in-memory
status at the same time.

It does not coordinate between Fargate tasks.

It does not protect EFS.

It does not make RAM global.

## The Global File Lock

The cross-process guard is the index-download file lock:

```text
/root/.astrometry/.index-download.lock
```

This file sits next to the shared index directory. Because Fargate mounts EFS
at `/root/.astrometry/data`, every Fargate task sees the same parent directory
and therefore the same lock file.

The same model is used locally because Docker Compose bind-mounts the host's
`${HOME}/.astrometry/data` to `/root/.astrometry/data`.

When an app instance finds that index files are missing, it tries to acquire an
exclusive `flock` on the lock file. Only one process can hold that lock at a
time.

The sequence is:

1. Process A checks the index directory and sees missing files.
2. Process A acquires the file lock.
3. Process B also sees missing files and tries to acquire the same lock.
4. Process B blocks while Process A holds the lock.
5. Process A downloads the missing files.
6. Process A releases the lock.
7. Process B acquires the lock, rechecks the directory, sees the files are now
   present, and skips the duplicate download.

This is the part that makes simultaneous Fargate startups safe.

## Readiness Check Details

The readiness preparation function does this:

1. Fetches the upstream directory listing from `INDEX_URL`.
2. Extracts the expected `index-*.fits` filenames.
3. Counts which expected files already exist locally and are non-empty.
4. If all files are present, marks this process ready.
5. If files are missing, acquires the global file lock.
6. Rechecks the files after acquiring the lock.
7. Downloads only missing files.
8. Recounts the files.
9. Marks this process ready if all expected files are now present.

The current completeness check is intentionally simple: an expected file counts
as present if it exists and has size greater than zero. It does not currently
deep-validate every FITS file.

## Sentinel File

After a successful check/download, the code writes:

```text
/root/.astrometry/data/.cat_astrometry_ready.json
```

This records that a successful readiness pass happened, along with counts and
timestamps. The core readiness decision still comes from checking the expected
index files and the current process's in-memory status.

## Route Behavior

`/health` returns the current process's readiness status. It does not force a
new expensive check by itself.

`/reset` starts a forced readiness check in the background for the current app
process.

`/astrometry` first checks the current process's in-memory readiness flag. If
that flag is false, it returns `503 not_ready` with the current readiness
status. It does not run the expensive file check on every request.

This keeps normal `/astrometry` requests cheap once the process has marked
itself ready.

## Failure Behavior

If the background preparation fails, the current process records:

```text
state=error
ready=false
error=<exception message>
```

Then `/health` exposes that error and `/astrometry` continues returning
`503 not_ready`.

The app process itself does not crash just because the background readiness
thread failed.

## Mental Model

There are two levels of coordination:

```text
Within one Python process:
  threading.RLock protects the in-memory status dictionary.

Across Fargate tasks or containers:
  flock on /root/.astrometry/.index-download.lock protects the shared
  index directory.
```

The readiness state in RAM is deliberately cheap and local. The file lock and
shared mounted directory are what prevent duplicate or overlapping downloads.
