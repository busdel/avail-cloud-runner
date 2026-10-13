# Data security &amp; trust model

**Zero telemetry.** The AVAIL software contains no analytics, no tracking,
no accounts and no phone-home of any kind. Every outbound network call it
can make is listed below, together with what is sent and how to switch it
off. Audited for v1.0.3; the audit is one `grep` away (see *Verify it
yourself*).

## Complete outbound-call inventory

| Component | When it fires | What leaves the machine | How to disable |
|---|---|---|---|
| **LLM advisor / assistant** (`avail_omics/llm.py`) | Only if *you* configure your own API key (`avail-omics set-key`) | Your prompts plus the aggregate result context you ask about (tables, metric summaries — not raw subject rows, unless you paste them) | Don't set a key; or `AVAIL_OFFLINE=1`; or `--no-llm` where supported |
| **PubMed literature** (`avail_omics/pubmed.py`) | Only when the advisor's literature step runs | Biomarker/feature **names** and disease-context terms (e.g. a bacterial genus). No patient identifiers, no tables | `AVAIL_OFFLINE=1` (helpers then return empty lists) |
| **Schaefer atlas download** (nilearn, imaging/radiomics) | Only when **no local atlas copy exists** | Nothing is sent; a public file is fetched from the nilearn data mirror | Bundle `atlas/schaefer_2018/` (included in the runner bundle) or `AVAIL_OFFLINE=1` |
| **Package installs** (`pip`) | First run of the local runner, or your own `pip install` | Package names to PyPI | Pre-install dependencies; the runtime pipeline itself installs nothing |

Nothing else — there is no other network code path in the package.

## Where your data lives, per deployment tier

| Tier | Data on your machine | Data elsewhere |
|---|---|---|
| **Local runner** (`run_me.py`) | Everything: raw images, features, models, results | Nothing. Uploading the small `results/` zip back to the web platform for the AI report is **optional** |
| **Self-hosted platform** | Everything (you run the server) | Nothing, unless you enable the LLM advisor/PubMed **with your own key**; the same inventory above applies |
| **Your own cloud runner** (GitHub Actions in *your* account) | Local preprocessing as you prefer | The uploaded data ZIP, inside your own GitHub account |
| **Hosted demo** (maintainer instance) | — | Only what you upload and choose to run; the hosted instance **never executes jobs on maintainer machines**; comparisons/agent read aggregate results only |

## Derived-data by design

The framework consumes *feature tables* (one row per subject), not raw
patient records. For the strictest governance, compute features inside
your own boundary (the imaging adapter, radiomics module and generic
loader all run locally) and share/upload only derived tables.

## Offline mode

```bash
export AVAIL_OFFLINE=1     # true/yes/on also accepted
avail-omics run-generic --data-root mydata --out results
```
Effects: LLM calls raise an explicit error, PubMed helpers return empty
results (advisor continues without literature), atlas downloads are
refused unless a local bundle exists. For fully offline imaging, keep
`atlas/schaefer_2018/` next to the pipeline (the runner bundle ships it).

## Verify it yourself

```bash
# 1) audit every network reference in the package
grep -rn "urllib.request\|http://\|https://" $(pip show avail-omics |
  awk '/Location/{print $2}')/avail_omics/ | grep -vE 'localhost|127\.0\.0\.1|eutils|api\.|generativelanguage|openrouter'
# expected: only the provider endpoints themselves and docstrings

# 2) confirm offline mode
AVAIL_OFFLINE=1 python -c "from avail_omics.pubmed import search_pubmed; \
print(search_pubmed('parkinson'))"      # -> []

# 3) reproduce a release artifact
sha256sum avail_omics-1.0.3-py3-none-any.whl
# compare with the hash shown on the GitHub release page
```

## Licence & governance

MIT-licensed, source public. As a user you are the data controller at
your tier: the maintainers receive nothing from your deployment and
operate no service that touches your data. The optional hosted demo is a
convenience layer with the same inventory above; nothing runs on
maintainer hardware.
