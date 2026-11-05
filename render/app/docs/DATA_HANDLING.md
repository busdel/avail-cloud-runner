# Data handling for the shared AVAIL interface

This page describes the boundary between the **shared hosted interface** and
**sensitive or identifiable data**.

## Use the shared instance only for:
- the fixed, de-identified demo;
- synthetic or fully de-identified tabular data that you are allowed to upload
  to a third-party host.

## Do not upload to the shared instance:
- patient-identifiable data, PHI, or data covered by a DUA/IRB that does not
  permit third-party hosting;
- raw clinical files whose loss or disclosure would require incident handling.

## Recommended paths for sensitive data

1. **Local runner (recommended).** Download the runner from a job page and run
   `python run_me.py mydata --out results` on your own machine. The package is
   installed into a private venv; raw data does not leave your machine. Upload
   only the derived results if your policy allows it.
2. **Self-hosted interface.** Run the same web app on your own server behind
   HTTPS, with `AVAIL_ALLOW_SERVER_RUN=1` if you want server-side execution.
3. **CLI/package.** `pip install avail-omics` and run the pipeline locally or on
   an institutional HPC system. Data never transits a shared host.

## What the shared instance stores

- Uploaded ZIPs are unpacked for validation and job execution.
- Validation summaries, logs, figures, tables and result ZIPs remain in the
  job directory until the job is deleted.
- The LLM advisor receives aggregate validation/results context, not the raw
  uploaded tables. Reviewer text is never used for training.
- Cloud-runner data URLs use an unguessable secret and are intended for the
  configured runner only.

## Deleting data

The owner of a job can open the job page and choose **Delete job & data**.
This permanently removes the job directory, including its uploaded data,
logs, figures, reports and result ZIP. Demo jobs are shared and cannot be
deleted from a reviewer session. There is no automatic expiry guarantee on a
self-hosted instance; operators should reset or purge storage according to
their own retention policy.
