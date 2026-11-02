# PhytoReg

PhytoReg is a two-stage plant regulatory model with a **shared spatial pathway and task-adaptive readouts**. It predicts signal, peaks, positional profiles and window totals across 37 species and 1,356 experimental tracks.

The first stage encodes DNA and estimates regulatory activity. The second stage combines these predictions with spatial context and frozen sequence features. Histone, OCR and TF share the spatial pathway and have task-specific readouts. `peak_logits` is the complete model's final peak prediction.

## Install

```bash
git clone https://github.com/caixudong2001/PhytoReg.git
cd PhytoReg
python -m pip install .
```

Python 3.10+, PyTorch and NumPy are required.

## Model weights

Place a complete trained two-stage bundle in `weights/PhytoReg/`. It contains `model.pt`, `config.json`, `tracks.json`, `metadata.json` and `manifest.json`. Both stages are loaded from this one directory; loading verifies all checksums.

```bash
phytoreg inspect weights/PhytoReg
```

## Python inference

```python
import torch
from phytoreg import PhytoReg

model = PhytoReg.from_pretrained('weights/PhytoReg', device='cuda')
tracks = model.tracks_for('arabidopsis_thaliana', task='Histone')[None]

# dna: int64 [B,131072], A/C/G/T/N = 0/1/2/3/4
# region: known inference-time bool mask [B,2048], independent of labels
with torch.inference_mode():
    outputs = model(dna, tracks, position_valid_mask=region)
```

DNA, tracks and masks must be on the model's device. One-hot DNA `[B,131072,5]` is also accepted. Predictions cover the central 65,536 bp in 2,048 bins of 32 bp. A batch can contain different species using padded track lists; each sequence's active tracks belong to one species.

| Output | Shape | Meaning |
|---|---|---|
| `signal_z` | `[B,2048,K]` | TRAIN-standardized continuous signal |
| `peak_logits` | `[B,2048,K]` | Final shared-spatial, task-adaptive peak logits |
| `profile_logits` | `[B,2048,K]` | Position logits for the signal distribution |
| `total_z` | `[B,K]` | Predicted log1p window total |
| `profile_score` | `[B,2048,K]` | Stabilized, unnormalized profile scores |

Apply sigmoid to `peak_logits` to obtain peak scores in `[0,1]`. `model.signal_in_registry_space(outputs['signal_z'], tracks)` reverses TRAIN standardization while retaining the registry's original transform and normalization.

`position_valid_mask` supplies known genomic support for spatial context, profile normalization and total pooling. Omitting it uses every output position.

## Command line

```bash
phytoreg predict weights/PhytoReg --fasta genome.fa --bed regions.bed --species arabidopsis_thaliana --task Histone --output predictions --batch-size 1 --format both --device cuda
phytoreg predict weights/PhytoReg --input inputs.npz --output predictions.npz --device cuda
phytoreg tracks weights/PhytoReg --species arabidopsis_thaliana --output tracks.tsv
```

Every command loads the complete model. FASTA output includes coordinate tables, per-track bedGraph and whole-window totals. NPZ input contains `dna`, `track_indices`, and optional `track_valid_mask` and `position_valid_mask`. Output paths must be new. See the [inference guide](docs/inference.md) for examples, coordinates, masks and units.

## Complete model bundle

```text
weights/PhytoReg/
  model.pt
  config.json
  tracks.json
  metadata.json
  manifest.json
```

`model.pt` includes both stages, all fixed context parameters and normalization buffers. Loading requires only this package and the five-file bundle. `model.train()` trains the spatial/task readout while keeping the first stage frozen and in evaluation mode. `model.save_pretrained('new_directory')` saves the complete model without optimizer state.

See the [model description](MODEL_CARD.md), [architecture and API](docs/architecture.md), and [contributing](CONTRIBUTING.md). Code is distributed under the [MIT License](LICENSE).
