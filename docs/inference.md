# FASTA/BED inference

## Install and load the model

```bash
git clone https://github.com/caixudong2001/PhytoReg.git
cd PhytoReg
python -m pip install .
```

Use a complete trained two-stage bundle. Put `model.pt`, `config.json`, `tracks.json`, `metadata.json` and `manifest.json` in `weights/PhytoReg/`. The weight file contains both stages, including fixed context priors and task-adaptive readouts. Every load verifies all bundle checksums.

```bash
phytoreg inspect weights/PhytoReg
phytoreg tracks weights/PhytoReg --species arabidopsis_thaliana --output tracks.tsv
```

`inspect` lists supported species. `tracks` lists track IDs, registry indices, tasks and available target/tissue annotations. Filter with `--task Histone`, `--task OCR` or `--task TF`; repeat `--task` to select multiple tasks. Species and chromosome identifiers are case-sensitive. Use a reference assembly appropriate to the experimental tracks.

## Run the included example

The included FASTA/BED files demonstrate the inference workflow.

```bash
phytoreg predict weights/PhytoReg --fasta examples/reference.fa --bed examples/regions.bed --species arabidopsis_thaliana --track-id arabidopsis_thaliana_0014 --output example_predictions --batch-size 1 --format both --device cpu
```

Use `--device cuda` or `--device cuda:0` for a GPU. Default device is CPU, default batch size is 1, and computation is FP32. The output directory must not exist.

## Predict real regions or a complete reference

```bash
phytoreg predict weights/PhytoReg --fasta genome.fa.gz --bed regions.bed --species oryza_sativa --task Histone --task OCR --output rice_predictions --batch-size 2 --format both --device cuda
```

Omit `--bed` to cover every FASTA sequence. Reduce `--batch-size` if CUDA reports out-of-memory. `--quiet` suppresses progress messages.

FASTA/BED can be plain text or gzip-compressed with a `.gz` suffix. The FASTA identifier is the first whitespace-delimited token after `>`. Wrapped/unwrapped lines, lowercase bases and IUPAC ambiguity codes are supported. Ambiguities other than A/C/G/T/N map to N and are counted in `run.json`. Empty records, duplicate identifiers and invalid DNA characters are rejected.

FASTA is streamed into a temporary seekable store. RAM does not scale with genome length; temporary disk space requires approximately one byte per reference base, including for compressed input. `--temp-dir` selects an existing directory for this store, which is closed automatically. GPU and prediction-array memory scale with batch size and selected track count. Coordinate results are written incrementally.

### Select exact tracks

Repeat `--track-id` for a few outputs, or use `--tracks-file selected_tracks.txt` containing one ID per line. Blank lines and `#` comments are accepted. This file is an ID list, not the `tracks.tsv` table. Duplicate/unknown IDs and IDs outside the chosen species/tasks are rejected. Requested ID order is preserved; otherwise catalog order is used. Without task/ID filters, all tracks of the selected species are predicted.

### BED coordinates and strand

BED3–6 is supported: `chrom start end [name [score [strand]]]`. Coordinates are **zero-based, half-open**. Intervals must satisfy `0 <= start < end <= chromosome_length`, with exact FASTA chromosome names. BED12 blocks are not interpreted; provide BED3–6 intervals. Blank lines, comments and `track`/`browser` headers are accepted.

Names and strand annotations are copied to `regions.tsv`. Prediction always uses the **forward reference**, including BED rows annotated `-`; strand annotations do not trigger reverse complementation. Overlapping/adjacent requests are merged. Each genomic bin is emitted once per selected track. Associate results with original, potentially overlapping features by genomic overlap with `regions.tsv`.

## Window and bin coordinates

The model reads 131,072 bp and predicts the central 65,536 bp in 2,048 bins of 32 bp. Output windows start at chromosome position zero and advance by 65,536 bp. Missing context at chromosome ends is padded with N. The fixed grid gives a genomic bin the same sequence context regardless of which overlapping BED request selected it.

Only bins intersecting the requested interval union are reported. They retain complete 32-bp coordinates, clipped only at the chromosome end. BED `chr1 33 70`, for example, reports `[32,64)` and `[64,96)`, with `requested_bp` of 31 and 6. Predictions describe model bins and are not interpolated to BED boundaries. `valid_bp` records real reference bases; the final chromosome bin may contain fewer than 32 bases.

Results use FASTA contig names and zero-based, half-open coordinates, in FASTA contig order with increasing bin starts within each track/contig. There is no chromosome aliasing or assembly conversion.

## Profile/total masks and units

By default, profile and total include bins overlapping real reference, excluding wholly padded output bins. Actual N bases remain valid positions.

`--valid-bed valid_regions.bed` optionally defines allowed spatial-context and profile/total positions. A bin is valid when all its real bases are covered by the union of valid intervals. This mask applies to the **complete model output window**, independently of the BED subset selected for reporting. Backbone features and continuous signal are unaffected. Spatial peak context uses this support; outside it, peaks retain their first-stage value.

- `signal_z`: TRAIN-standardized continuous signal.
- `signal_registry`: reverses TRAIN standardization; original registry transform/normalization remains. Definitions are in the bundle's `tracks.json`.
- `peak_logit`: final two-stage shared-spatial, task-adaptive peak score.
- `peak_probability`: sigmoid of the final `peak_logit`.
- `profile_probability`: profile scores normalized over **all valid bins of the complete output window**; zero at invalid positions. A reported BED subset need not sum to one.
- `total_z`: log1p total for the **complete model output window** under its validity mask. All-invalid windows report `NA` totals and zero profile probabilities.

## Output files

| File | Contents |
|---|---|
| `run.json` | Completion state, input hashes, model checksum, conventions, counts and runtime |
| `tracks.tsv` | Selected track IDs, indices, annotations and bedGraph filename prefixes |
| `regions.tsv` | Original requested intervals and name/strand annotations |
| `chrom.sizes` | FASTA contig names and lengths |
| `windows.tsv` | Window/context coordinates, padding, valid-bin and reported-bin counts |
| `totals.tsv.gz` | One complete-window total per track, with coordinates and validity |
| `predictions.tsv.gz` | One row per emitted bin/track; default `--format tsv` |
| `bedgraph/` | Per-track signal/peak/profile tracks; `--format bedgraph` or `both` |

Prediction-table columns are `chrom`, `start`, `end`, `window_id`, `bin_index`, `registry_index`, `track_id`, `signal_z`, `signal_registry`, `peak_logit`, `peak_probability`, `profile_logit`, `profile_probability`, `profile_valid`, `valid_bp`, `requested_bp`.

bedGraph names use registry indices, such as `track_00000.signal_registry.bedGraph.gz`. Map them to biological tracks with `tracks.tsv`. Invalid positions are omitted from profile bedGraph files. Numeric values use nine significant digits to preserve float32 precision.

Successful runs end with `"status": "COMPLETED"` in `run.json`. Failed inference records `FAILED` and an error; its files are partial. Validation failures before output creation leave no result directory. Existing results are never overwritten; retry with a new directory.

### Optional bigWig conversion

bedGraph output needs no extra binary dependency. If UCSC's `bedGraphToBigWig` utility is installed, this Linux/macOS example converts a track:

```bash
gzip -dc rice_predictions/bedgraph/track_00000.signal_registry.bedGraph.gz | LC_ALL=C sort -k1,1 -k2,2n > signal.sorted.bedGraph
bedGraphToBigWig signal.sorted.bedGraph rice_predictions/chrom.sizes signal.bw
```

Replace the track index with one in your output catalog. Coordinates follow the [UCSC BED specification](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) and [bedGraph specification](https://genome.ucsc.edu/goldenPath/help/bedgraph.html).

## Pre-encoded NPZ input

```bash
phytoreg predict weights/PhytoReg --input inputs.npz --output predictions.npz --batch-size 2 --device cuda
```

Inputs are `dna`, `track_indices`, and optional `track_valid_mask`/`position_valid_mask`, with the same nonempty batch dimension. Outputs retain the five model arrays. NPZ has no genomic coordinate metadata. It batches GPU computation but keeps input/output arrays in CPU memory; use streamed FASTA output for genome-scale work.
