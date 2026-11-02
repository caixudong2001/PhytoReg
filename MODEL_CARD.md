# PhytoReg model

PhytoReg is a two-stage plant regulatory model with a shared spatial pathway and task-adaptive readouts. It predicts continuous signal, peaks, positional profiles and window totals for Histone, open chromatin (OCR) and transcription-factor (TF) tracks.

## Architecture

The first stage uses a convolutional/Transformer DNA backbone to produce sequence features and regulatory activity estimates. Signal, initial peak, profile and total have independent per-track readouts.

The second stage combines initial peak logits and continuous signal through shared multiscale spatial filters. Assay-condition gates select spatial scales, and task-specific readouts produce the final peak logits. The TF readout also uses frozen sequence features. Context priors, calibration coefficients and normalization buffers are included in the complete model.

The first-stage backbone and activity heads remain frozen during second-stage training. The shared spatial pathway and task-adaptive feature readout are trainable.

## Inputs and outputs

The model supports 37 species and 1,356 experimental tracks. It reads 131,072 bp and predicts the central 65,536 bp in 2,048 bins of 32 bp.

Inputs are DNA, registry track indices and optional position/track masks. The model returns `signal_z`, `peak_logits`, `profile_logits`, `profile_score` and `total_z`. `peak_logits` contains the final two-stage prediction.

## Python interface

The `PhytoReg` class loads both stages from one model directory:

```python
from phytoreg import PhytoReg

model = PhytoReg.from_pretrained('weights/PhytoReg', device='cuda')
```

The complete bundle contains `model.pt`, `config.json`, `tracks.json`, `metadata.json` and `manifest.json`.

See [architecture and API](docs/architecture.md) and the [inference guide](docs/inference.md). Code is distributed under the [MIT License](LICENSE).
