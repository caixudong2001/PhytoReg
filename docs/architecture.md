# Architecture and API

## Stage one: sequence and regulatory activity

The DNA backbone embeds bases, applies seven convolutional downsampling stages, nine RoPE/SwiGLU transformer layers and a U-Net decoder. It averages features into 32-bp bins and retains the central 65,536 bp, with 512 features per position.

Signal, initial peak, profile and total have independent LayerNorm/512-to-256 projections and per-track weight tables. Initial peaks supply the second stage internally. Signal combines two affine terms including species-specific feature normalization and per-track scale. TRAIN label normalization is stored in buffers.

Total uses masked mean sequence features and a learned log1p valid-length term. Profile scores exponentiate logits after subtracting the maximum over valid positions. Normalize their sum for a positional distribution; an empty region gives zero scores.

## Stage two: shared spatial pathway and task-adaptive readout

Inputs are first-stage peak logits and standardized signal. Fixed local features use mask-normalized means over 3, 9 and 33 bins. A shared 1-by-1 stem feeds four branches with dilation 1, 4, 16 and 64, local pooling and symmetric five-tap kernels. A shared mixing layer combines central and contextual features. The released pathway width is 64.

Known assay-condition embeddings gate spatial scales. Histone, OCR and TF each have spatial readout weights, a bias and a central-input term. TF also uses a LayerNorm/512-to-256 feature projection and registered-track readout over frozen sequence features. Peak-unavailable tracks retain their first-stage peak output.

Fixed learned context pathways and feature references define residual priors and calibration. Their parameters and coefficients are part of the complete model. There is one trainable shared spatial pathway and one task-adaptive feature readout; inference loads no external reference model or fitting artifact.

The final prediction is `peak_logits`. Signal, profile and total come from the frozen first stage. Spatial support uses known inference regions and track availability, independently of observed labels. Invalid spatial positions retain first-stage peak values; padded tracks are zero.

## Input contract

- DNA is integer `[B,input_length_bp]`, A/C/G/T/N = 0/1/2/3/4, or one-hot `[B,input_length_bp,5]`. An all-zero one-hot base is a masked token, distinct from N.
- `track_indices` is int64 `[B,K]`. Use registry indices from `tracks_for(species, task)` or `tracks.json`, including noncontiguous values.
- `track_valid_mask` is boolean `[B,K]`. Omitted masks mark nonnegative indices active. Explicitly masked entries can use arbitrary padding sentinels.
- Each sequence's active tracks share one species normalization group; batch rows can contain different species.
- `position_valid_mask` is boolean `[B,L]`, controlling spatial context, profile support and total pooling. It does not change DNA encoding or continuous signal.
- `from_features(h, track_indices, ...)` accepts float `[B,L,512]` and executes both readout stages. Features are detached for second-stage fitting.

Inference masks describe known regions. Label-specific validity masks belong in losses and metrics, not in model inputs.

## Configuration and complete weights

`PhytoRegConfig` contains `backbone`, `head`, `spatial` and `region_total_length`. `SpatialConfig` specifies shared-pathway width, fixed-context width and the number of assay conditions. `configs/phytoreg.json` describes the released architecture.

The five-file `PhytoReg` bundle contains `model.pt`, `config.json`, `tracks.json`, `metadata.json` and `manifest.json`. The tensor state dictionary includes both stages, fixed context parameters, registry mappings and normalization buffers. The catalog binds each track to its species, task, condition and availability. Loading verifies SHA256 checksums, catalog mappings and strict state keys.

```python
from phytoreg import PhytoReg

model = PhytoReg.from_pretrained('weights/PhytoReg', device='cuda')
model.save_pretrained('new_directory')
```

The destination must not exist. Saving includes both stages and excludes optimizer state.

## Training and precision

`model.train()` enables shared spatial and task-adaptive feature readouts. The backbone, first-stage heads, fixed context pathways and references retain frozen parameters and evaluation-mode statistics. Use a peak loss and an optimizer over parameters with `requires_grad=True`; signal/profile/total are preserved during updates.

Model parameters and readouts use FP32.
