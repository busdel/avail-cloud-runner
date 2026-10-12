#!/bin/bash
# Assemble the deployment bundle for the HF Space (or any Docker host).
# Result: deploy/hf-space/app/  (push together with Dockerfile, entrypoint.sh,
# README.md to the Space repository root).
set -eu
SRC=/mnt/limbic/AVAIL_GPU
DST="$SRC/deploy/hf-space/app"
rm -rf "$DST"
mkdir -p "$DST"

rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/avail_ai" "$DST/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/webapp" "$DST/"
rsync -a "$SRC/run_generic.py" "$SRC/run_llm_advisor.py" "$DST/"
rsync -a "$SRC/docs" "$DST/"

mkdir -p "$DST/experiments/results"
rsync -a --include '*.py' --include 'THEORY_NOTES.md' --include 'METHODS_NOTES.md' \
      --exclude '*' "$SRC/experiments/" "$DST/experiments/"
for d in wallen_avail_v2 ad_avail hmp2_avail metacardis_results metacardis_cc/out; do
    mkdir -p "$DST/experiments/results/$d"
    rsync -a "$SRC/experiments/results/$d/meta_benchmark_results.tsv" \
          "$DST/experiments/results/$d/" 2>/dev/null || true
done
rsync -a "$SRC/experiments/results/reconstruction_benchmark_vascellari.tsv" \
      "$DST/experiments/results/" 2>/dev/null || true

mkdir -p "$DST/data_demos" "$DST/uploads"
if [ -f "$SRC/data_demos/ds004392_excerpt.zip" ]; then
    rsync -a "$SRC/data_demos/ds004392_excerpt.zip" "$DST/data_demos/"
fi

echo "bundle ready: $DST"
du -sh "$DST"
