# Download

## Python package

```bash
pip install --upgrade avail-omics
pip install "avail-omics[deep]"    # neural and deep-tabular meta-learners
```

- PyPI: https://pypi.org/project/avail-omics/
- Source: https://github.com/busdel/avail-cloud-runner
- Gallery: https://busdel.github.io/avail-cloud-runner/viz/

## Docker

```bash
docker build -t avail-demo .
docker run --rm -p 7860:7860 avail-demo
```

Then open http://localhost:7860.

## Local runner

The package and local runner keep raw data on your machine. The public demo is
only for de-identified or synthetic data.

## Documentation

- Input format: `docs/DOCUMENTATION.md`
- Tutorial: `docs/TUTORIAL.md`
- R interoperability: https://busdel.github.io/avail-cloud-runner/docs/R_INTEROP.md
