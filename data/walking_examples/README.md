# Walking Examples

Small, real video examples converted into this repo's clip format.

## Included

- `raw_videos/walking-bradford-5s.mp4`
- `raw_videos/walking-kobuleti-5s.mp4`
- `clips/*.npz` extracted tensors with key `frames`
- `index.csv` ready for `scripts/train.py` / `scripts/test.py`

## Source Attribution

The clips are short derivatives from Wikimedia Commons walking videos:

- `Bradford Town Hall Square (1896)`  
  `https://commons.wikimedia.org/wiki/File:Bradford_Town_Hall_Square_(1896).webm`
- `Kobuleti2025ispani-walking`  
  `https://commons.wikimedia.org/wiki/File:Kobuleti2025ispani-walking.webm`

Refer to each file page for full license details.

## Regenerate Clip Tensors

```bash
python scripts/extract_video_examples.py \
  --video-dir data/walking_examples/raw_videos \
  --output-root data/walking_examples \
  --overwrite
```

This writes:

- `data/walking_examples/clips/*.npz`
- `data/walking_examples/index.csv`

## Train/Test Example

```bash
python scripts/train.py \
  --config configs/train_examples.yaml \
  --data-root data/walking_examples \
  --run-name walking-example
```

```bash
python scripts/test.py \
  --config configs/train_examples.yaml \
  --data-root data/walking_examples \
  --checkpoint artifacts/walking-example/checkpoints/best_weights.npz \
  --split test
```
