"""Batched inference over FASTA references, optional BED requests, or NPZ arrays."""
from __future__ import annotations

from itertools import islice
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .config import PhytoRegConfig
from .genome import (BIN_SIZE_BP, FastaReference, IntervalIndex, Region, count_windows, encode_dna,
                     open_text, parse_region, plan_windows, read_bed, window_arrays)
from .model import file_sha256
from .prediction_output import PredictionWriter, write_json


def select_tracks(model, species=None, tasks=None, track_ids=None):
    return _select_tracks(model.track_catalog, species, tasks, track_ids)


def _select_tracks(records, species, tasks, track_ids):
    catalog = []
    for original in records:
        record = dict(original)
        record.setdefault('track_id', str(record['registry_index']))
        catalog.append(record)
    selected = [r for r in catalog if (species is None or r['species_id'] == species)
                and (not tasks or r['task_group'] in tasks)]
    if track_ids is not None:
        if len(set(track_ids)) != len(track_ids):
            raise ValueError('Track IDs must not be duplicated')
        by_id = {r['track_id']: r for r in selected}
        missing = [name for name in track_ids if name not in by_id]
        if missing:
            raise ValueError(f'Track IDs do not match the selected species/tasks: {missing[:5]}')
        selected = [by_id[name] for name in track_ids]
    if not selected:
        raise ValueError('No tracks match the requested species/tasks/track IDs; use phytoreg tracks to list them')
    return selected


def read_track_ids(path):
    with open_text(path) as handle:
        names = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith('#')]
    if not names:
        raise ValueError('Track list is empty')
    return names


def _batches(items, size):
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _requested_regions(sizes, bed, region):
    if region is not None:
        if bed is not None:
            raise ValueError('--bed and --region are mutually exclusive')
        return [parse_region(*region, sizes=sizes, source='--region')]
    if bed is not None:
        return read_bed(bed, sizes)
    return [Region(chrom, 0, length) for chrom, length in sizes.items()]


def plan_genome(model_directory, *, fasta, species, bed=None, region=None, valid_bed=None,
                tasks=None, track_ids=None, output_format='tsv'):
    """Count genomic outputs using the model configuration and track catalog."""
    directory = Path(model_directory)
    config = PhytoRegConfig.from_json(directory / 'config.json')
    records = json.loads((directory / 'tracks.json').read_text(encoding='utf-8'))
    required = {'registry_index', 'species_id', 'task_group'}
    if not isinstance(records, list) or any(not isinstance(row, dict) or required - row.keys() for row in records):
        raise ValueError('Invalid tracks.json catalog')
    tracks = _select_tracks(records, species, tasks, track_ids)
    with FastaReference(fasta, store_bases=False) as reference:
        sizes = reference.sizes
        requests = IntervalIndex(_requested_regions(sizes, bed, region))
        if valid_bed is not None:
            read_bed(valid_bed, sizes)
    windows = count_windows(requests, config.backbone.downstream_output_length_bp)
    bins = count_windows(requests, BIN_SIZE_BP)
    peak_tracks = sum(bool(row.get('peak_enabled', True)) for row in tracks)
    return dict(status='PLANNED', species=species, tracks=[row['track_id'] for row in tracks],
                output_format=output_format, windows_planned=windows, bins_planned=bins,
                prediction_rows=bins * len(tracks) if output_format in ('tsv', 'both') else 0,
                total_rows=windows * len(tracks),
                bedgraph_rows_upper_bound=(bins * (2 * len(tracks) + peak_tracks)
                                           if output_format in ('bedgraph', 'both') else 0))


def predict_genome(model, *, fasta, output, species, bed=None, region=None, valid_bed=None, tasks=None,
                   track_ids=None, batch_size=1, output_format='tsv', temp_dir=None, quiet=False):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if output_format not in ('tsv', 'bedgraph', 'both'):
        raise ValueError('format must be tsv, bedgraph or both')
    directory = Path(output)
    if directory.exists():
        raise FileExistsError('Output directory already exists; choose a new path')
    tracks = select_tracks(model, species, tasks, track_ids)
    input_length = model.config.backbone.input_length_bp
    output_length = model.config.backbone.downstream_output_length_bp
    flank = (input_length - output_length) // 2
    started = time.monotonic()
    with FastaReference(fasta, temp_dir) as reference:
        sizes = reference.sizes
        regions = _requested_regions(sizes, bed, region)
        requests = IntervalIndex(regions)
        valid_regions = IntervalIndex(read_bed(valid_bed, sizes)) if valid_bed is not None else None
        total_windows = count_windows(requests, output_length)
        provenance = {name: dict(file=Path(path).name, sha256=file_sha256(path))
                      for name, path in [('fasta', fasta), ('bed', bed), ('valid_bed', valid_bed)] if path is not None}
        status = dict(status='RUNNING', model='PhytoReg', model_sha256=model.metadata.get('loaded_weights_sha256'),
            inputs=provenance, species=species, tracks=[r['track_id'] for r in tracks],
            input_length_bp=input_length, output_length_bp=output_length, bin_size_bp=BIN_SIZE_BP,
            coordinates='zero-based half-open; reference-forward; contig-origin grid',
            bed_selection='emit complete 32-bp bins intersecting the union of requested intervals; clip at contig end',
            position_mask='bins fully covered by valid_bed' if valid_bed else 'bins overlapping real reference sequence',
            profile_normalization='all valid positions of each complete model output window',
            peak_readout='shared spatial pathway and task-adaptive readouts over known valid positions',
            signal_units='TRAIN-standardized and registry-normalized; not raw coverage',
            total_units='log1p total per complete model output window',
            ambiguous_iupac_bases_mapped_to_N=reference.ambiguous_bases,
            batch_size=batch_size, device=str(model.device), precision='float32', output_format=output_format,
            windows_planned=total_windows, windows_completed=0, emitted_bins=0)
        if region is not None:
            requested = regions[0]
            status['region'] = dict(chrom=requested.chrom, start=requested.start, end=requested.end)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / 'run.json', status)
        model.eval()
        try:
            with PredictionWriter(directory, tracks, sizes, regions, output_format) as writer, torch.inference_mode():
                indices = torch.tensor([r['registry_index'] for r in tracks], device=model.device, dtype=torch.long)
                windows = plan_windows(sizes, requests, output_length)
                for batch in _batches(windows, batch_size):
                    arrays = [window_arrays(w, output_length, requests, valid_regions) for w in batch]
                    tokens = np.stack([encode_dna(reference.fetch(w.chrom, w.start - flank,
                                      w.start - flank + input_length)) for w in batch])
                    dna = torch.from_numpy(tokens).to(model.device)
                    positions = torch.from_numpy(np.stack([a[4] for a in arrays])).to(model.device)
                    track_indices = indices[None].expand(len(batch), -1)
                    predictions = model(dna, track_indices, position_valid_mask=positions)
                    if not all(torch.isfinite(value).all() for value in predictions.values()):
                        raise ValueError('Model produced non-finite predictions')
                    predictions['signal_registry'] = model.signal_in_registry_space(predictions['signal_z'], track_indices)
                    predictions['peak_probability'] = predictions['peak_logits'].sigmoid()
                    scores = predictions['profile_score']
                    predictions['profile_probability'] = scores / scores.sum(1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
                    if not all(torch.isfinite(value).all() for value in predictions.values()):
                        raise ValueError('Output scaling produced non-finite predictions')
                    cpu = {key: value.cpu().numpy() for key, value in predictions.items() if key != 'profile_score'}
                    for i, window in enumerate(batch):
                        status['emitted_bins'] += writer.write(window, arrays[i], {k: v[i] for k, v in cpu.items()}, input_length, output_length)
                        status['windows_completed'] += 1
                    if not quiet and (status['windows_completed'] <= batch_size
                                      or status['windows_completed'] % (10 * batch_size) == 0
                                      or status['windows_completed'] == total_windows):
                        print(f"Predicted {status['windows_completed']}/{total_windows} windows", file=sys.stderr, flush=True)
                    write_json(directory / 'run.json', status)
            status.update(status='COMPLETED', elapsed_seconds=round(time.monotonic() - started, 3))
            write_json(directory / 'run.json', status)
            return status
        except BaseException as exc:
            status.update(status='FAILED', error=str(exc), elapsed_seconds=round(time.monotonic() - started, 3))
            write_json(directory / 'run.json', status)
            raise


def predict_npz(model, source, output, batch_size=1):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Output file already exists; choose a new path')
    with np.load(source, allow_pickle=False) as data:
        if not {'dna', 'track_indices'} <= set(data.files):
            raise ValueError('Input NPZ requires dna and track_indices')
        allowed = {'dna', 'track_indices', 'track_valid_mask', 'position_valid_mask'}
        if set(data.files) - allowed:
            raise ValueError(f'Unknown inputs: {sorted(set(data.files) - allowed)}')
        arrays = {key: data[key] for key in data.files}
    if any(array.ndim == 0 for array in arrays.values()):
        raise ValueError('NPZ arrays require a batch dimension')
    count = len(arrays['dna'])
    if count == 0 or any(len(array) != count for array in arrays.values()):
        raise ValueError('NPZ arrays must share a nonempty batch dimension')
    parts = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            batch = {key: torch.from_numpy(value[start:start + batch_size]).to(model.device) for key, value in arrays.items()}
            predictions = model(**batch)
            if not all(torch.isfinite(value).all() for value in predictions.values()):
                raise ValueError('Model produced non-finite predictions')
            for key, value in predictions.items():
                parts.setdefault(key, []).append(value.cpu().numpy())
    result = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, **result)
    return {key: list(value.shape) for key, value in result.items()}
