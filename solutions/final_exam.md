# Final Exam — Theoretical Questions

## 3.1 API Design

### Question 1

The `POST /diagnose` request schema contains exactly 7 fields:

1. `piece_id` (string) — identifies the piece in the response
2. `die_matrix` (int) — selects the correct row from `reference_times.json`
3. `lifetime_2nd_strike_s` (float | null) — cumulative time at 2nd strike
4. `lifetime_3rd_strike_s` (float | null) — cumulative time at 3rd strike
5. `lifetime_4th_strike_s` (float | null) — cumulative time at 4th strike
6. `lifetime_auxiliary_press_s` (float | null) — cumulative time at auxiliary press
7. `lifetime_bath_s` (float | null) — cumulative time at quench bath

**Why cumulative timestamps, not pre-computed partial times?**
The PLC records absolute clock timestamps relative to the start of each forging cycle. The upstream system (the sensor layer) naturally produces cumulative values — it does not subtract them. Accepting cumulative input means the API matches the raw sensor contract exactly, with no transformation required before calling the endpoint. Pre-computing partial times before the API call would push business logic out of the service and into the caller, which is the wrong layer.

**Why is this the minimum set?**
The 5 partial segment times are derived from exactly these 5 cumulative values using the formulas in §1.5: `furnace_to_2nd_strike = lifetime_2nd_strike_s` (absolute), and each subsequent partial is a difference of two adjacent cumulative values. There is no prior timestamp for the furnace step, so `lifetime_2nd_strike_s` acts as an absolute time. Removing any one of the 5 cumulative fields would make one or two partial times uncomputable. Adding `piece_id` and `die_matrix` makes the response self-describing and enables reference lookup. No other fields are needed.

### Question 2

`reference_times.json` is read once in the FastAPI `lifespan` context manager at container startup and stored in the module-level `reference_times` dict. Every call to `POST /diagnose` reads from this in-memory dict.

**Why this is the right approach for a containerized deployment:**
A containerized API is designed to handle many requests per second. If `reference_times.json` were opened and parsed on every request, each call would incur a filesystem read, a JSON parse, and a dict allocation — all unnecessary work, since the file never changes while the container is running. Loading once at startup means the cost is paid exactly once during the container's lifespan, and all requests share the same in-memory object. This is the 12-factor app principle applied to configuration: treat config as part of the environment, load it at boot, keep it in memory. It also means the API fails fast at startup (with a clear file-not-found error) rather than failing silently on the first request.

## 3.2 Containerization And Deployment

### Question 1

```dockerfile
FROM python:3.13-slim
```
Uses the official slim Python 3.13 image as the base. `slim` omits development tools and documentation, keeping the image small. Python 3.13 matches the `requires-python` in `pyproject.toml`.

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
```
Copies the `uv` binary from its official image into the container. This avoids installing uv via pip, which would be slower and add a bootstrapping dependency. The binary is self-contained and requires no runtime prerequisites.

```dockerfile
WORKDIR /app
```
Sets the working directory for all subsequent instructions and the runtime process. All relative paths in `COPY` and `CMD` resolve against `/app`.

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
```
Copies only the dependency manifest and lockfile first, then installs dependencies. Because Docker caches layers by the content hash of their inputs, this layer is only rebuilt when `pyproject.toml` or `uv.lock` changes — not when application code changes. `--frozen` ensures the exact versions from the lockfile are used. `--no-dev` excludes pytest from the production image. `--no-install-project` skips installing the project itself as a package (there is no wheel to install).

```dockerfile
COPY src/ ./src/
COPY reference_times.json ./
```
Source code and data are copied after dependencies. Changes to `src/` or `reference_times.json` only invalidate from this layer onward, leaving the slow `uv sync` layer cached.

```dockerfile
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
`EXPOSE 8000` documents the container port. The `CMD` starts uvicorn using the project's virtual environment via `uv run`. `--host 0.0.0.0` is required inside a container — the default `127.0.0.1` would make the port unreachable from outside the container.

### Question 2

**Option 1 — AWS Lambda + API Gateway**

*Advantage:* Scales to zero when idle — no cost when there are no requests. For a low-traffic diagnostics endpoint that is only called occasionally, this is cheaper than Fargate, which bills continuously while the task is running.

*Disadvantage:* Lambda has a cold-start latency of several hundred milliseconds to a few seconds when the function has been idle. For a real-time diagnostics API where an operator is waiting on a response, this unpredictable latency spike is worse than Fargate's always-warm container. Lambda also has a 15-minute execution limit and a 250 MB deployment package limit, which could be restrictive if the model or dependencies grow.

**Option 2 — Amazon EC2 (virtual machine)**

*Advantage:* Full control over the runtime environment — OS version, system libraries, network configuration, and instance type. Useful if the API had native dependencies that are difficult to containerise, or if the team needed to run other processes on the same host.

*Disadvantage:* Requires managing the OS: applying security patches, configuring auto-scaling, and handling instance failures manually. Fargate abstracts all of this — there are no instances to patch or replace. For a stateless API container, EC2's extra operational overhead provides no benefit over Fargate.

## 3.3 Testing And Extensibility

### Question 1

The exam requires testing `diagnose()` directly as a pure function rather than through HTTP requests to a running API for three reasons:

**Speed and simplicity:** Calling `diagnose(piece_dict, refs)` directly runs in microseconds with no process startup, no port binding, no HTTP stack, and no async event loop. The 34 tests in `tests/test_diagnose.py` complete in under a second. An HTTP-based test suite would require starting a uvicorn process, waiting for it to be ready, and tearing it down — adding seconds of overhead and infrastructure noise.

**Isolation:** A pure function test verifies only the business logic — the delay-detection rules, cause mapping, and null propagation. It does not accidentally test FastAPI routing, request parsing, or response serialisation at the same time. When a test fails, the failure points directly at the logic, not at a networking issue or framework behaviour.

**Portability:** `diagnose()` can be imported and called in any Python environment — a notebook, a CLI script, or a test — without starting a server. This is why the exam specifies keeping the diagnosis logic out of the route handler. If the logic were written directly inside the FastAPI handler, it would be untestable without an HTTP client.

### Question 2

To support a new die matrix `6001`, every layer of the stack needs one targeted change:

**1. Data file — `reference_times.json`**
Add a new key `"6001"` with the 5 median partial times computed from production data for that matrix. This is the only data change required.

**2. Code — no changes needed**
`diagnose()` and `app.py` are both matrix-agnostic: they look up the matrix key dynamically from `reference_times.json`. No hardcoded matrix IDs exist in the logic. The unknown-matrix 400 error will no longer trigger for `6001` once the key is present in the file.

**3. Tests — add 6 new unit tests**
Add `6001` to the `MATRICES` list in `test_diagnose.py`. The parametrized test functions (`test_all_ok`, `test_furnace_to_2nd_penalized`, etc.) automatically generate 6 new test cases for the new matrix — one per scenario. No new test code needs to be written.

**4. Validation artifacts — optional update**
`validation_pieces.csv` and `validation_expected.json` do not need to change for correctness, but it is good practice to add a row exercising matrix `6001` (e.g. an all-OK piece and one penalized piece) to keep the golden test representative.

**5. Redeployment**
Rebuild the Docker image (which re-bundles `reference_times.json`), push the new image to ECR with a new version tag, and update the ECS task definition to reference the new image URI. The ECS service will deploy the new revision with zero downtime if a service with desired-count ≥ 1 is used.
