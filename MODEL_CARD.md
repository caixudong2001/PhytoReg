# PhytoReg model card

## Model

PhytoReg is a multi-species plant regulatory model with a shared DNA backbone and four independent track readouts: continuous signal, peak classification, positional profile and window total. It supports 37 species and 1,356 experimental tracks: Histone 724, OCR 301 and TF 331. TF tracks cover Arabidopsis thaliana and Oryza sativa.

The model has 204,779,587 parameters. Standard input is 131,072 bp; prediction covers the central 65,536 bp at 32-bp resolution. The full species and track identities are included in the model bundle.

## Parameters and training

The released parameter set is the retained 2,100-step checkpoint from a full-backbone and independent-readout continuation experiment. The underlying backbone had previous supervised training; this is not a model trained from random initialization in only 2,100 steps. The continuation used seed 17, 2,000 training windows per species, effective batch 48, and trained the backbone and readouts together. Training was stopped by the author.

Packaging as PhytoReg changes the model format and public API, with no additional training. The tensor values are preserved. No language-model output head or track-query generator is used at inference.

## Evaluation scope

The last completed full 37-species benchmark belongs to step 2,000. Its scores must not be presented as a new benchmark of the released 2,100-step checkpoint. Signal Pearson and peak average precision are separate metrics and must be reported separately. Species-averaged gains do not imply improvement of every track or every species.

Verification found exact equality of all 235 parameter/buffer tensors, all five returned outputs and feature gradients over all 37 species, and native DNA inference for Arabidopsis, rice and grape. Maximum absolute differences were zero. This checks the correctness of packaging; it does not establish performance on an external dataset.

## Intended use and limitations

Use the model for plant regulatory sequence research on the listed species and measured track definitions. Independent output tables cannot generate a new assay/tissue track from metadata. Extrapolation to unrepresented species or experimental conditions requires separate validation.

The model's signal units are transformed and standardized. Peak probabilities are scores, not experimentally calibrated binding probabilities. Position masks and track-specific valid labels affect comparisons. The reference evaluation uses noncoding/valid-region masks for profile/total outputs and supervision; applying all-position masks changes those outputs.

## Release information

Source code is available in the [PhytoReg repository](https://github.com/caixudong2001/PhytoReg) under the [MIT License](LICENSE). The repository contains code, configuration and tests; trained weights and training data are distributed separately. Their availability and redistribution terms are not established by this code repository.
