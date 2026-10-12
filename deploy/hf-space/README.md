---
title: AVAIL-AI
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8090
pinned: false
license: mit
---

# AVAIL-AI — web interface (stable hosted instance)

Availability-aware multi-omics integration & learning. Upload a ZIP ->
automatic validation + AI data-quality feedback -> run on the free cloud
(GitHub Actions) or on your own computer -> AI decision report +
publication figures.

The hosted instance **never executes heavy jobs on the hosting
infrastructure**: `AVAIL_ALLOW_SERVER_RUN=0` is enforced by the image.
Compute paths are the free cloud runner, the local runner bundle, or the
`avail-omics` Python package (https://pypi.org/project/avail-omics/).

## Space secrets (Settings -> Variables and secrets)

| Secret | Required | Purpose |
|---|---|---|
| `AVAIL_WEB_PASSWORD` | yes | login password for the interface |
| `AVAIL_GH_REPO` | optional | `owner/repo` of your cloud runner (free cloud runs) |
| `AVAIL_GH_TOKEN` | optional | fine-grained token, Actions: write on that repo only |
| `AVAIL_LLM_API_KEY` | optional | assistant agent + AI reports (OpenAI-compatible) |
| `AVAIL_LLM_API_KEY_GROQ` | optional | automatic fallback provider |

## Caveats

- Storage is ephemeral: uploads/results reset on rebuild or restart.
  Download or re-upload results as needed.
- Data hygiene: cloud runs upload the data ZIP to GitHub Actions in the
  runner account configured above. Prefer the local runner (`RUN ON YOUR
  OWN COMPUTER` in the app, or `pip install avail-omics`) for sensitive data.
- See `DEPLOY.md` in the repository for alternative stable-hosting options
  (ngrok static domain, named Cloudflare tunnel, VPS).
