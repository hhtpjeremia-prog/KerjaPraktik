# Current run reproducibility snapshot

This folder contains a compact, GitHub-tracked snapshot of the current NewStyle run: active manual annotations, split and ROI metadata, Grounding DINO proposal/mask manifests, GT and NB03-NB06 metrics/summaries, plus the YOLO and CNN inference weights.

## Moving to another device

1. Clone/pull this repository to get code, metadata and the two close-set model weights.
2. Copy the full `output/` tree from the synchronized Google Drive project folder into the project root. The full tree contains raw/curated frames, crops, segmentation-mask PNGs, Grounding DINO checkpoints, large NB06 videos and frame/track tables. Those are intentionally kept in Drive because together they are many gigabytes and make GitHub clones unwieldy.
3. Keep the relative directory layout under `output/`. The metadata snapshots here are for provenance and quick inspection; the pipeline expects its working artifacts under `output/`.

The snapshot does not include `output/` backups, raw videos, large overlays/event clips, every per-frame tracking row, or the 657 MiB Grounding DINO safetensors checkpoint. Retrieve those from Drive if needed; do not rerun training to recreate them.
