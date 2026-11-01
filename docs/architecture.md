# Architecture and API

## Sequence backbone

The NTv3-style PyTorch backbone embeds DNA bases, applies seven convolutional downsampling stages, nine RoPE/SwiGLU transformer layers, and a U-Net decoder. It averages full-resolution features into 32-bp bins and retains the central 65,536 bp. Each output position has 512 features. The backbone retains standard self-attention Q/K/V operations.

## Independent readouts

Each of the four tasks has a separate LayerNorm and 512→256 projection. Each registered track has a directly learned 256-dimensional weight vector and bias for each readout. Track IDs index these parameter tables. No metadata encoder, generated track query, track-query attention, or language-model output head is instantiated.

The signal head has two learned affine terms sharing its projection. One uses species-normalized features and a per-track scale. Both terms are part of this model's signal readout. The stored feature normalization and TRAIN label normalization are buffers. Their order is retained in FP32 so the released parameters preserve their numerical predictions.

Peak and profile are independent position-wise logits. Total uses the masked mean of backbone features, its own projection/readout, and a learned log1p valid-length term. Profile scores use exponentiation after subtracting the per-track maximum over valid positions. Normalize their sum to obtain a position distribution when at least one position is valid; an empty mask gives all-zero scores.

The output tables cover a fixed set of experimental tracks. Metadata cannot generate outputs for an unregistered track. Within one sequence, requested tracks must share a species normalization group.

## Input contract

- DNA is an integer tensor `[B,input_length_bp]` with A/C/G/T/N encoded as 0/1/2/3/4, or one-hot `[B,input_length_bp,5]`. In one-hot input, an all-zero base is a dynamic masked token and is distinct from N.
- `track_indices` is int64 `[B,K]`. Registry indices may contain gaps and differ from contiguous head rows. Use `tracks_for(species, task)` or `tracks.json`.
- `track_valid_mask` is boolean `[B,K]`. Omitted masks mark nonnegative indices as active. Masked padding does not contribute to outputs.
- `position_valid_mask` is boolean `[B,L]`. It controls profile support and total pooling, not backbone feature extraction or signal/peak validity. Downstream metrics still need their label-specific validity masks.
- `from_features(h, track_indices, ...)` accepts float features `[B,L,512]` with the same readout masks.

## Configuration and checkpoints

`PhytoRegConfig` contains only the sequence backbone, independent heads and valid-length term. Shape validation rejects invalid widths and bin boundaries. `configs/phytoreg.json` describes the full released model.

The on-disk format is `PhytoReg`. `model.pt` contains a tensor state dictionary, `config.json` its architecture, `tracks.json` its track catalog, and `metadata.json` its model-level description. The manifest identifies the format and SHA256 of each file. Loading uses PyTorch's restricted tensor loader and strict state-key matching.

The model remains trainable through standard PyTorch. Keep parameters in FP32; an external autocast context may accelerate the backbone while readouts run in FP32. The release includes inference weights, not optimizer or random-generator state for an exact training resume.

For deterministic FP32 reproduction on CUDA, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization, disable TF32 and cuDNN benchmarking, set cuDNN deterministic mode, and enable `torch.use_deterministic_algorithms(True)`. Cross-device or reduced-precision arithmetic need not be bitwise identical.
