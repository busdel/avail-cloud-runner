---
title: AVAIL-omics demo
sdk: docker
app_port: 7860
---

# AVAIL-omics quick demo

MLP-style demo: upload a generic AVAIL ZIP, validate it, run three fast
meta-learners, and download a results ZIP. Documentation, tutorial, download
and about tabs are included.

```bash
docker build -t avail-demo .
docker run --rm -p 7860:7860 avail-demo
```

Open http://localhost:7860.

This is a demo, not a medical device. Do not upload identifiable patient data.
