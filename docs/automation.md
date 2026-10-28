# Continuous verification

Keep results for committed kernels up to date. Actions runs the existing verifier
and report scripts; S3 stores evidence; the `results` branch serves the catalog
and reports to the website. Source and policy changes go through PRs.
Verification and publication are implemented. AI below is a proposed extension.

## When work starts

The workflow is [continuous-verification.yml](../.github/workflows/continuous-verification.yml).

| Trigger | Execution and publication |
| --- | --- |
| Contributor opens, updates or reopens a PR | Contract tests, then verification and report previews; no publication credentials. |
| Maintainer pushes or merges to `main` | Contract tests; when automation is enabled, run and publish. |
| GitHub daily schedule, 04:17 UTC | When enabled, compare current inputs with published results and fill gaps. This is a scheduled time, not a start-time guarantee. |
| Maintainer runs the workflow manually | One kernel or the whole catalog; `force` requests verification. Preview by default; publication also requires enabled `main` and `publish: true`. |

PR/push events ignore changes confined to `docs/**`, `README.md`, `LICENSE` or
`assets/**`. `results` commits do not trigger verification. Once the snapshot
loader is deployed, page loads read `latest.json` and that snapshot without a redeploy.
New PR runs cancel older runs for that PR. Main, scheduled and manual runs do not
cancel each other; only their publication jobs share a concurrency group.

## How a run works

After contracts pass, `prepare` builds the pinned runtime, fixes the checkout SHA
and compares each kernel with the published snapshot. It discovers
`examples/<name>/input.json` paired with `corpus/<name>.ptx`; adding a kernel
does not require editing the workflow.

| Change or saved state | Planner decision |
| --- | --- |
| PTX, input, reference or binding changed | Verify that kernel. |
| Verifier, shared reference helpers, runtime or verification limits changed | Verify affected identities, normally the full catalog. |
| Only report skill or publication code changed | Reuse matching verifier output; recompose and check reports. |
| Missing output or tool failure | Try verification again. |
| Matching output, including findings, timeout or unsupported results | Reuse it unless forced; a daily run does not repeat an unchanged timeout. |
| Known memory demand exceeds the budget | Record `deferred_resource`; `force` does not raise the budget. |

Identity covers input hashes, verifier code, the built native binary, Python and
system packages, and verification limits. Shard count does not affect it.
Tasks outside a manual selection stay `pending` unless matching output is reusable.

The plan supplies the shard matrix, with at most four hosted runners at once.
Shards run kernels serially, with 500 seconds and 8 GiB per verifier.
The process group has a separate deadline. Each networkless container limits the
whole process tree to 10 GiB, two CPUs and 256 processes. Reference Python runs
inside it; export rejects symlinks and special files after the container stops.

`report` collects records after the shards finish, including partial failures.
The existing composer/checker produces reports. Failed reports are withheld;
verifier output remains available.

Job success, kernel verdict and report status are separate. `current`/`reused`
mean output is available, including inconclusive results. `pending`/`deferred_resource`
have no current verdict; `failed` means execution or collection failed. Earlier
conclusions stay under `historical`; existing performance studies keep their inputs.

## Publication and recovery

S3 stores immutable evidence under content hashes. The `results` branch stores
`runs/<run_id>/<snapshot_id>/` with `results.json`, `manifest.json` and reports;
`latest.json` points to one snapshot. Each kernel record binds its inputs,
runtime, budgets, execution state and report/evidence references.

The publisher rechecks records/reports and uploads/checks evidence before committing
the snapshot and pointer together. Publishers run serially, check current `main`
and the baseline, and refuse to discard completed results. Git updates are never forced.
GitHub can replace a waiting publisher with a newer one; the group is not a queue
that guarantees every run will publish. The next dispatch compares saved results again.

| Interruption | Recovery |
| --- | --- |
| Worker or report fails | Keep other records; the next plan retries missing/failed verification and recomposes reports. |
| Upload or Git update fails | The old pointer stays usable. Retry with retained artifacts; already uploaded objects are reusable. |
| Publication repeats | An already published snapshot is a no-op. A successful retry can produce a new snapshot under the same run. |
| Source or baseline changed while work ran | Stop the stale publication; a new dispatch plans against current state. |
| Unpublished artifacts expire | Rerun the missing work. The daily planner reads published results, not old Actions artifacts. |

Intermediate artifacts last 14 days; the worker image lasts two. After image
expiry, start a new dispatch. Published evidence has no automatic expiry.
Jobs pass the plan, image and reports by artifact ID; missing uploads fail explicitly.
Raw shards retain stable names so a failed-shard retry can use the other shards' output.
For rollback, disable publication, stop active publishers and restore a validated
`latest.json`. We save results, not a durable queue of unfinished jobs.

## Adding AI later

Publish base results first. Trial AI manually on one kernel in a published
snapshot before enabling automatic runs. Skip clean results unless requested.

1. Prepare matching inputs, output, source and the existing report skill. Pin the
   CLI/model/skill versions; defer if the material exceeds the context budget.
2. Claude returns JSON in the skill's `draft.json` format. Treat source/comments
   as data. Give the model no execution/edit tools or GitHub/storage write credentials.
   [`--bare`](https://code.claude.com/docs/en/cli-reference) skips configuration
   discovery; tool restrictions and process isolation must be set separately.
3. Run composer/checker, then a separate Claude session reviewing guards,
   participants, source reasoning and completion relationships against the same
   evidence. Allow at most one revision and another review.
4. Trusted code publishes only checked, reviewed explanations, with the draft,
   review and usage record as evidence. Create a new snapshot only if the original
   is still current; otherwise leave the current snapshot alone and replan.

Reuse an explanation only when evidence hashes, skill/composer rules, model
configuration and review policy match; recompose facts and rerun the checker.
AI cannot change the raw verdict or inputs. Suggested fixes need a separate PR.
Reviewer approval is not another verifier result. Later, failure diagnosis or new
kernel proposals can produce ordinary PRs; merging them uses the same pipeline.

Before paid runs, trusted code must reserve calls in a persistent ledger using
conditional object-store writes: $10 per dispatch and $20 per UTC day, including
revisions. Recheck at day boundaries; retries keep their spending and unresolved
calls retain reservations. Bound turns/time and set the provider's spending limit:
CLI estimates are not a hard billing cap. None of these budget controls exists yet.
Failure or insufficient budget leaves the published base result available.

Before automatic use, test wrong citations, unsupported claims, reviewer rejection,
budget exhaustion, duplicate attempts and a newer snapshot arriving during review.

## Why these choices

| Public implementation | What applies here |
| --- | --- |
| [vLLM job definitions](https://github.com/vllm-project/vllm/blob/0beab6c251e8046c0c6cf2fef114cd23b58cda74/.buildkite/test_areas/kernels.yaml) and [selection rules](https://github.com/vllm-project/ci-infra/blob/5fa07cf6cbc6e90cb72abdebc43cb6288f343932/buildkite/pipeline_generator/README.md) declare dependencies, hardware and time limits. | Keep task selection explicit. Our kernel inputs are already enumerable, so content identities suffice; no per-kernel workflow or Buildkite generator is needed. |
| [gh-aw's CI coach](https://github.com/github/gh-aw/blob/a74a1e171d5a52aec4dbd3a3f09e7d1dafda783c/.github/workflows/ci-coach.lock.yml) separates the agent job from jobs that write to GitHub. | Keep AI output as data for the existing publisher to check. We need this permission boundary, not its workflow compiler. |
| [Temporal](https://github.com/temporalio/documentation/blob/main/docs/encyclopedia/activities/activity-definition.mdx#idempotency) records completed activity results and requires retry-safe side effects. | Keep the recovery limit above explicit. Reconsider a durable scheduler when jobs need checkpoints or several executors must claim work across failures. |

A server can use the same task/record format, but needs a plan for its actual
runtime and budgets. Add it when deferred kernels need execution. Upstream imports
and dependency/model upgrades remain reviewed source changes.

## Enable and check

Require PR review on protected `main` before enabling publication. The environment's
branch restriction only checks the ref; it does not enforce review of workflow code.

Set repository variables:

| Variable | Value |
| --- | --- |
| `SENTRYWARD_AUTOMATION_ENABLED` | `true` enables automatic runs and manual publication from `main`. |
| `SENTRYWARD_S3_ENDPOINT` | HTTPS S3 API endpoint, without bucket or query parameters. |
| `SENTRYWARD_S3_BUCKET` | Evidence bucket. |
| `SENTRYWARD_EVIDENCE_URL` | Public HTTPS prefix serving the bucket's `objects/` keys. |
| `AWS_REGION` | Storage signing region; default `us-east-1`. |

Restrict the `automation-publish` environment to `main`. Supply
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and optional `AWS_SESSION_TOKEN`
with read/conditional-write access to `objects/`. Only the publisher gets these
credentials and a GitHub token that can update `results`.

Storage must support `If-None-Match: *`, permanent public object URLs and browser
GETs from `https://sentryward.io` through CORS. Uploads are checked through both
authenticated and public endpoints. Start with a small manual preview, then
publish from enabled `main` with `publish: true`. Deploy the website snapshot
loader only after the first valid snapshot exists; it has no floating-main fallback.

`python -m tools.automation --help` exposes `plan`, `verify`, `report` and
`publish`, plus runtime/baseline inspection. Plan from a clean committed checkout.
Run public contracts with `python3 -m unittest discover -s tests/automation`;
they use small Git fixtures and fake verifier/storage clients. Production also
needs a Linux container preview and a check of the configured storage/publisher.
