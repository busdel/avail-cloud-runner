## Bring your own compute (BYO): GPU tiers and trust model

The hosted interface never executes jobs on maintainer machines. Compute
happens on infrastructure you control, in one of four tiers. Each tier
keeps data inside a boundary you choose.

### Tier 1 - Local runner (recommended start)
Download the runner bundle from any job page (or `pip install avail-omics`)
and run on your own computer, CPU or GPU:

    python run_me.py mydata.zip --out results
    # or: avail-omics run-generic --data-root my_data --out results

Upload only the derived `results/` back to the interface for the AI report.
**Trust:** nothing leaves your machine except the aggregates you choose to
upload.

### Tier 2 - Self-hosted web platform
Run the same interface on your own server:

    export AVAIL_WEB_ROOT=/path/to/AVAIL_GPU
    export AVAIL_WEB_PASSWORD=choose-a-strong-password
    export AVAIL_ALLOW_SERVER_RUN=1        # your machine is now the server
    waitress-serve --port=8090 --call webapp.app:create_app

Expose it however you like (VPN, SSH tunnel, or an outbound-only quick
tunnel such as `cloudflared tunnel --url http://localhost:8090`).
**Trust:** your infrastructure, your data, your access control; no
maintainer has an account on it.

### Tier 3 - Your own cloud runner (GitHub Actions)
Fork the runner repository, create a **fine-grained token scoped to that
one repository** (Actions: read/write only), and point the platform at it:

    export AVAIL_GH_REPO=your-account/your-runner
    export AVAIL_GH_TOKEN=github_pat_...

Job dispatch then goes to **your** GitHub account's runner.
**Trust:** code and data flow to your account; keep uploads
aggregate-only if data is sensitive.

### Tier 4 - Pull-based worker agent (roadmap)
A small worker daemon for user GPU servers: it connects **outbound only**
(no inbound ports), registers with a per-worker, short-lived token, and
pulls jobs from the queue into a per-job container sandbox with CPU/GPU/
memory limits. Nothing is exposed; the server can revoke the worker at any
time. Planned for a future release.

### Security checklist (any tier)
- Outbound-only connections; never open inbound GPU ports to the internet.
- Isolate execution: container or virtualenv, per-job temp directory,
  resource limits; do not reuse environments across tenants.
- Tokens: fine-grained, per-service, short-lived, rotated; least privilege.
- Data hygiene: derive features first; upload aggregates unless you own
  the infrastructure end to end.
- Provenance: pinned dependencies, reproducible seeds, checksums of the
  code bundle; the integrity audit script is included and re-derivable.
- The web platform itself: password + rate limiting, TLS via the tunnel,
  static file serving restricted to its own directory, fail-closed input
  validation.
