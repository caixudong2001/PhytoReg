# PhytoReg

PhytoReg predicts plant regulatory activity from DNA across 37 species and 1,356 experimental tracks. It uses a shared sequence backbone and independent signal, peak, profile and total readouts. The model has approximately 205 million parameters.

## Install

```bash
git clone https://github.com/caixudong2001/PhytoReg.git
cd PhytoReg
python -m pip install .
```

Python 3.10+, PyTorch and NumPy are required. The package provides one model class, `PhytoReg`, and one model bundle format, `PhytoReg`.

## Load and predict

The trained model bundle is supplied separately. Place its five files in `weights/PhytoReg/`. For FASTA/BED input, automatic windowing, batched prediction and genomic-coordinate output:

```bash
phytoreg predict weights/PhytoReg --fasta genome.fa --bed regions.bed --species arabidopsis_thaliana --task Histone --output predictions --batch-size 1 --format both --device cuda
```

Omit `--bed` to cover the entire FASTA. Results include compressed coordinate tables, per-track bedGraph files and complete-window totals. The [inference guide](docs/inference.md) includes a runnable example, track selection, masks, units and output schemas.

For direct Python inference:

```python
import torch
from phytoreg import PhytoReg

model = PhytoReg.from_pretrained('weights/PhytoReg', device='cuda')
tracks = model.tracks_for('arabidopsis_thaliana', task='Histone')[None]

# dna: int64 [batch, 131072], A/C/G/T/N = 0/1/2/3/4
# region: bool [batch, 2048], positions included in profile/total readouts
with torch.inference_mode():
    outputs = model(dna, tracks, position_valid_mask=region)
```

`dna`, `tracks` and masks must be on the model's device. One-hot DNA `[B,131072,5]` is also accepted. The central 65,536 bp is predicted at 32-bp resolution, giving 2,048 output bins. The model uses registry indices returned by `tracks_for`; each sequence's selected tracks belong to one species. A batch can contain different species with padded track lists.

| Output | Shape | Meaning |
|---|---|---|
| `signal_z` | `[B,2048,K]` | Continuous signal in TRAIN-standardized units |
| `peak_logits` | `[B,2048,K]` | Peak classification logits; apply sigmoid for probabilities |
| `profile_logits` | `[B,2048,K]` | Position logits for the signal distribution |
| `total_z` | `[B,K]` | Predicted log1p window total |
| `profile_score` | `[B,2048,K]` | Stabilized, unnormalized profile scores; invalid positions are zero |

`model.signal_in_registry_space(outputs['signal_z'], tracks)` reverses TRAIN standardization. Registry-level normalization remains; its per-track transform, center, scale and clipping bounds are described in `tracks.json`. This output should not be interpreted directly as raw coverage.

`position_valid_mask` selects positions for total pooling and profile normalization. Omitting it uses all positions. The released model was trained/evaluated with the recorded noncoding/valid-region mask; use the same mask to reproduce that evaluation. The mask does not modify DNA backbone features. `track_valid_mask` handles padding in `[B,K]` track indices.

## Command line

```bash
phytoreg inspect weights/PhytoReg
phytoreg tracks weights/PhytoReg --species arabidopsis_thaliana --output tracks.tsv
phytoreg predict weights/PhytoReg --input inputs.npz --output predictions.npz --device cuda
```

The NPZ input contains `dna`, `track_indices`, and optionally `track_valid_mask` and `position_valid_mask`. Existing prediction files are not overwritten. Bundle checksums are verified on every load.

## Model bundle

```
weights/PhytoReg/
  model.pt
  config.json
  tracks.json
  metadata.json
  manifest.json
```

The bundle contains the complete backbone, four output heads and normalization buffers. Loading requires only this package and bundle. `model.train()` enables training of the backbone and heads; `model.save_pretrained('new_directory')` writes the same format without optimizer state.

See [architecture and API](docs/architecture.md), [model card](MODEL_CARD.md), and [contributing](CONTRIBUTING.md). Source code is distributed under the [MIT License](LICENSE).
