# Continuous verification

The workflow verifies changed kernels, generates reports from the saved results,
and publishes an immutable snapshot on the `results` branch. Generated files stay
out of `main`. The website reads the snapshot through `latest.json`.

This first version produces deterministic reports using the existing report skill.
It does not call a model. AI explanations, cost accounting and server execution
are separate follow-up work.

## What runs

The planner discovers committed `examples/<name>/input.json` files paired with
`corpus/<name>.ptx`, plus each case's reference and binding when present.

| Change | Work |
| --- | --- |
| PTX, input, reference or binding | Verify the affected kernel. |
| Verifier, shared reference helpers, dependencies or verification options | Verify all affected task identities. |
| Report skill | Reuse matching verification evidence and regenerate reports. |
| Ordinary documentation | Preserve cached verification. |

The identity includes input hashes, verifier sources, the actual native binary,
Python and system packages, and resource limits. Reports are always recomposed
and checked against their frozen input and result files. Exit code zero does not
mean verification passed; the verifier's result and diagnostics remain intact.

PRs produce credential-free checks and previews. Enabled `main` pushes publish
results; a daily run reconciles unfinished work against the latest snapshot.
Manual dispatch selects one kernel or the full catalog, can force verification,
and defaults to preview only.

Four hosted runners execute kernels serially within each shard. Each kernel gets
500 seconds and an 8 GiB verifier budget. Its container has no network, a read-only
filesystem, a 10 GiB total memory limit, two CPUs and at most 256 processes.
Reference Python runs only inside that container. After the container stops,
the host rejects symlinks and special files before exporting artifacts.

Execution status is separate from the verifier's conclusion:

| State | Meaning |
| --- | --- |
| `current` | This task produced a verifier result, including inconclusive outcomes. |
| `reused` | Matching saved evidence was retrieved and checked. |
| `pending` | Outside the manual selection. |
| `deferred_resource` | Recorded memory demand exceeds the runner budget. |
| `failed` | Execution or evidence collection failed. |

Historical timeout cases are scheduled. High-memory cases remain explicitly
accounted for rather than receiving an automatically enlarged budget. Historical
verification, synchronization-removal experiments and B200 measurements retain
their original source and input identities.

## Enable publication

Configure these repository variables:

| Variable | Value |
| --- | --- |
| `SENTRYWARD_AUTOMATION_ENABLED` | `true` enables automatic runs and manual publication from `main`. |
| `SENTRYWARD_S3_ENDPOINT` | HTTPS S3 API endpoint, without the bucket or query parameters. |
| `SENTRYWARD_S3_BUCKET` | Evidence bucket. |
| `SENTRYWARD_EVIDENCE_URL` | Public HTTPS prefix serving that bucket's `objects/` keys. |
| `AWS_REGION` | Storage signing region; defaults to `us-east-1`. |

Create an `automation-publish` GitHub environment restricted to `main`. Add
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, for temporary credentials,
`AWS_SESSION_TOKEN`. Grant this identity read and conditional-write access to
`objects/`. The job's `GITHUB_TOKEN` needs permission to update `results`.
Verification and reporting jobs receive no publication credentials.

Storage must support `If-None-Match: *` and serve permanent public object URLs.
For a base such as `https://evidence.example.org/sentryward`, the stored key
`objects/<sha256>/output.json` must be readable beneath that exact prefix.
Uploads are checked through both the authenticated and public endpoints.
Disable automatic expiration for published evidence and allow browser `GET`
requests from `https://sentryward.io` through CORS.

Start with a small manual preview while the enable flag is unset. After checking
its results, enable publication and dispatch from `main` with `publish: true`.
Deploy the website's snapshot loader after the first valid snapshot exists;
it reports a missing or invalid pointer rather than falling back to `main`.

## Snapshots and recovery

`run_id` identifies a dispatch. `snapshot_id` hashes its plan, records and evidence,
so a successful retry can publish a new snapshot without changing the earlier one:

```text
latest.json
runs/<run_id>/<snapshot_id>/results.json
runs/<run_id>/<snapshot_id>/manifest.json
runs/<run_id>/<snapshot_id>/reports/<kernel>/report.md
```

The publisher validates records and reports, uploads and checks all evidence,
then commits the snapshot and pointer together. Repeating the same publication is
idempotent. A late run cannot replace a newer source revision or discard a matching
completed result. If another dispatch changes the baseline during execution,
publication stops; a new manual or scheduled dispatch replans from that baseline.
Git updates are serialized and never forced. Publication does not create releases.

Actions artifacts retain intermediate work for 14 days; the worker image is kept
for two days. After it expires, start a new dispatch instead of retrying a verifier
job. Durable evidence lives in object storage. To roll back, disable automation,
stop in-flight publishers and commit a previously validated `latest.json` to
`results`, retaining all snapshots and objects.

## Development

`python -m tools.automation --help` exposes `plan`, `verify`, `report`, and `publish`,
plus runtime and baseline inspection. Stages share a plan and case-directory
layout. Plan from a clean committed checkout; source links pin that commit.
The container build fixes Ubuntu, system packages and CPU PyTorch dependencies.

Run the public contracts without credentials or third-party Python packages:

```sh
python3 -m unittest discover -s tests/automation
```

These tests use small Git fixtures and fake verifier/storage clients. Before
production enablement, also run a Linux container preview and check the configured
storage and publication identity. macOS address-space limits do not substitute
for the Linux resource-limit check.
