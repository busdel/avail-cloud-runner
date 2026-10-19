# AVAIL worker — connect YOUR own server, safely

The worker is a small job server you run on **your own machine** (GPU or
CPU). Any AVAIL interface (the web UI, a local copy, or curl) can submit a
job to it; the pipeline runs inside your infrastructure and only the
results travel back.

Nothing is exposed by default and every request needs a bearer token.


## Without pip: use the local-runner bundle

The downloadable runner bundle can serve jobs directly (no package install
commands needed; it sets up a private environment on first run):

```bash
python run_me.py --serve --port 8787
# prints a bearer token; binds to 127.0.0.1
```

Then open a job page, choose "Run on my own server", and paste the URL and
token. If the web interface runs elsewhere, expose the worker through your
own outbound tunnel and paste that URL instead.


## Quickstart (on your server)

```bash
pip install avail-omics
avail-omics worker-token          # creates ~/.config/avail/worker_token (0600) and prints it
avail-omics worker --name my-gpu  # serves on http://127.0.0.1:8787
```

Smoke test:

```bash
curl -s http://127.0.0.1:8787/health     # -> {"ok": true, "name": "my-gpu", ...}
```

## Connect from the web interface

Job page -> **"Run on my own server"** -> enter the worker URL and token.
The data ZIP is sent directly from that UI to your worker; the interface
polls your worker for status and downloads the results into the job page
(history, downloads, AI report and agent all keep working).

Note: if you submit from a hosted copy of the UI, your data first passes
through that UI's server. For sensitive data use your own local copy of
the interface (or the CLI) so nothing transits maintainer infrastructure.

## Expose it safely (outbound-only, no open ports)

Keep the worker bound to `127.0.0.1` and put a private tunnel in front:

```bash
# private: reachable only from your own tailnet
tailscale serve --bg 8787

# or public HTTPS (only if you need it): a quick tunnel
cloudflared tunnel --url http://localhost:8787
```

Both options connect **outbound** from your machine; there is no inbound
port to open on your router. Always use the token, preferably over HTTPS.

## API (all endpoints require `Authorization: Bearer <token>`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness (public) |
| POST | `/jobs?flags=robustness,ci&name=x` | submit a data ZIP (raw body) |
| GET | `/jobs/<id>` | status: queued/running/done/error (+ log tail) |
| GET | `/jobs/<id>/log` | full log |
| GET | `/jobs/<id>/results.zip` | results archive (when done) |
| DELETE | `/jobs/<id>` | delete the job and its sandbox |

curl example:

```bash
TOKEN=$(cat ~/.config/avail/worker_token)
curl -X POST -H "Authorization: Bearer $TOKEN" \
     --data-binary @mydata.zip "http://127.0.0.1:8787/jobs?flags=robustness"
```

## Security model

- **Token**: 32-byte URL-safe bearer token, compared in constant time;
  stored in `~/.config/avail/worker_token` (0600). Rotate by deleting the
  file and restarting the worker.
- **No default exposure**: binds to `127.0.0.1`; you choose the tunnel.
- **Sandboxing**: each job lives in its own directory under the workdir;
  the runner refuses ZIP entries that escape it; jobs run as your user
  (never root), with `nice` and an optional memory cap
  (`--mem-gb`, Linux `RLIMIT_AS`).
- **Concurrency**: one job at a time by default (`--max-parallel`),
  so a submission cannot starve your GPU.
- **Results**: served only with the token; delete with `DELETE /jobs/<id>`
  or by removing the workdir.

## Run as a service (optional)

```ini
# ~/.config/systemd/user/avail-worker.service
[Unit]
Description=AVAIL worker
[Service]
ExecStart=%h/anaconda3/bin/avail-omics worker --name my-gpu --mem-gb 32
Restart=on-failure
[Install]
WantedBy=default.target
```

`systemctl --user enable --now avail-worker`

## Current scope

- Runs the generic pipeline (`avail-omics run-generic`); the `robustness`
  flag additionally runs the leak-free permutation test.
- Other comparison modules (calibration, conformal, decision curves, ...)
  are not yet exposed through the worker — submit those from a self-hosted
  full interface or the package directly.
- No timeouts are imposed: long jobs are expected on training machines.