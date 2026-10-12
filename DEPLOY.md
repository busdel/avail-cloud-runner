# Deploying AVAIL-AI with a stable public URL

The hosted interface never executes heavy jobs on the hosting machine
(`AVAIL_ALLOW_SERVER_RUN=0` is enforced in the container image). Compute
stays on the free cloud runner, the local runner bundle, or the
`avail-omics` package.

Three stable-hosting options, easiest first.

---

## Option A — Hugging Face Space (Docker) · free · stable `*.hf.space`

1. Create a Hugging Face account (a project-specific account is fine) and
   a write token: https://huggingface.co/settings/tokens
2. Create a Docker Space: https://huggingface.co/new-space
   (name e.g. `avail-ai`, SDK = Docker, hardware = CPU basic).
3. Build the bundle and push:

```bash
bash deploy/build_space_bundle.sh

git clone https://huggingface.co/spaces/<user>/avail-ai
cd avail-ai
cp ../AVAIL_GPU/deploy/hf-space/{Dockerfile,entrypoint.sh,README.md} .
cp -r ../AVAIL_GPU/deploy/hf-space/app .
git add -A && git commit -m "AVAIL-AI web interface"
git push    # username: <user>, password: <HF write token>
```

4. In the Space settings, add secrets:

| Secret | Required | Purpose |
|---|---|---|
| `AVAIL_WEB_PASSWORD` | yes | login password |
| `AVAIL_GH_REPO` / `AVAIL_GH_TOKEN` | optional | your cloud runner (free cloud runs) |
| `AVAIL_LLM_API_KEY` (+`_GROQ`) | optional | assistant agent / AI reports |

5. Stable URL: `https://<user>-avail-ai.hf.space`

Notes: storage is ephemeral (uploads/results reset on restart); the free
tier sleeps after inactivity and wakes on the next visit (~30 s cold start).

---

## Option B — ngrok static domain · free · `*.ngrok-free.app`

1. Create an ngrok account (free) and claim the included static domain:
   https://dashboard.ngrok.com/domains
2. Run locally (web app on port 8090):

```bash
ngrok http --domain=<your-static>.ngrok-free.app 8090
```

3. Stable URL: `https://<your-static>.ngrok-free.app`

Notes: the free plan shows a one-time browser interstitial for visitors and
has monthly bandwidth limits; fine for reviews and light trials.

---

## Option C — Named Cloudflare tunnel + your domain · ~$10/year domain

1. Add your domain to Cloudflare (free plan), then:

```bash
cloudflared tunnel login
cloudflared tunnel create avail
cloudflared tunnel route dns avail avail.yourdomain.com
cloudflared tunnel run --url http://localhost:8090 avail
```

2. Stable URL: `https://avail.yourdomain.com`

Most durable and professional-looking option; requires owning a domain.

---

## Point the stable link at the new URL

After any option above:

- GitHub Pages landing (`busdel/avail-cloud-runner/index.html`): replace the
  meta-refresh and button URL with the new stable URL, and
- stop the quick-tunnel sync in `~/.local/bin/avail-tunnel-watch.sh`
  (keep the tunnel as a spare) or leave it unchanged if you keep using the
  quick tunnel as the primary.

## Security checklist (all options)

- Keep `AVAIL_ALLOW_SERVER_RUN=0`; no inbound ports on your machines beyond
  the tunnel process.
- Use secrets for the web password, GH token (single-repo scope) and LLM
  keys; never bake them into the image or repository.
- Storage is ephemeral on HF/ngrok hosts: patients' uploads should travel
  through the local-runner path instead of a public host.
