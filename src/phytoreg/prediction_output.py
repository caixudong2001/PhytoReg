"""Streaming, coordinate-aware prediction tables and per-track bedGraph files."""
from collections import OrderedDict
import csv
import gzip
import json
from pathlib import Path


TRACK_FIELDS = ['registry_index', 'track_id', 'species_id', 'task_group', 'target', 'tissue',
                'biosample', 'signal_enabled', 'peak_enabled', 'bedgraph_prefix']
BIN_FIELDS = ['chrom', 'start', 'end', 'window_id', 'bin_index', 'registry_index', 'track_id',
              'signal_z', 'signal_registry', 'peak_logit', 'peak_probability', 'profile_logit',
              'profile_probability', 'profile_valid', 'valid_bp', 'requested_bp']
WINDOW_FIELDS = ['window_id', 'chrom', 'output_start', 'output_end', 'model_output_end',
                 'input_start', 'input_end', 'pad_left_bp', 'pad_right_bp', 'valid_bins', 'reported_bins']
TOTAL_FIELDS = ['window_id', 'chrom', 'start', 'end', 'registry_index', 'track_id', 'total_z', 'valid_bins']


def write_tracks(handle, tracks):
    writer = csv.DictWriter(handle, TRACK_FIELDS, delimiter='\t', lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    for row in tracks:
        writer.writerow(dict(row, bedgraph_prefix=f"track_{row['registry_index']:05d}"))


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class BedGraphFiles:
    """Limit open streams even for species with hundreds of tracks."""
    def __init__(self, directory, max_open=24):
        self.directory = directory
        directory.mkdir()
        self.max_open = max_open
        self.open = OrderedDict()
        self.created = set()

    def write(self, index, kind, lines):
        key = (index, kind)
        if key not in self.open:
            if len(self.open) >= self.max_open:
                self.open.popitem(last=False)[1].close()
            path = self.directory / f'track_{index:05d}.{kind}.bedGraph.gz'
            self.open[key] = gzip.open(path, 'at' if key in self.created else 'xt', encoding='utf-8')
            self.created.add(key)
        self.open.move_to_end(key)
        self.open[key].writelines(lines)

    def close(self):
        for handle in self.open.values():
            handle.close()
        self.open.clear()


class PredictionWriter:
    def __init__(self, directory, tracks, sizes, regions, output_format):
        self.directory = Path(directory)
        self.tracks = tracks
        self.handles = []
        self.bedgraphs = None
        try:
            with (self.directory / 'tracks.tsv').open('x', encoding='utf-8', newline='') as handle:
                write_tracks(handle, tracks)
            with (self.directory / 'chrom.sizes').open('x', encoding='utf-8') as handle:
                for chrom, size in sizes.items():
                    handle.write(f'{chrom}\t{size}\n')
            with (self.directory / 'regions.tsv').open('x', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
                writer.writerow(['chrom', 'start', 'end', 'name', 'strand'])
                writer.writerows((r.chrom, r.start, r.end, r.name, r.strand) for r in regions)
            self.windows = self._table('windows.tsv', WINDOW_FIELDS)
            self.totals = self._table('totals.tsv.gz', TOTAL_FIELDS)
            self.bins = self._table('predictions.tsv.gz', BIN_FIELDS) if output_format in ('tsv', 'both') else None
            if output_format in ('bedgraph', 'both'):
                self.bedgraphs = BedGraphFiles(self.directory / 'bedgraph')
        except BaseException:
            self.close()
            raise

    def _table(self, name, fields):
        path = self.directory / name
        handle = (gzip.open(path, 'xt', encoding='utf-8', newline='') if name.endswith('.gz')
                  else path.open('x', encoding='utf-8', newline=''))
        self.handles.append(handle)
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(fields)
        return writer

    def write(self, window, arrays, values, input_length, output_length):
        starts, ends, valid_bp, requested_bp, valid = arrays
        selected = (requested_bp > 0).nonzero()[0]
        flank = (input_length - output_length) // 2
        input_start = window.start - flank
        input_end = input_start + input_length
        valid_count = int(valid.sum())
        self.windows.writerow([window.identifier, window.chrom, window.start, min(window.start + output_length, window.chrom_length),
            window.start + output_length, input_start, input_end, max(0, -input_start),
            max(0, input_end - window.chrom_length), valid_count, len(selected)])
        for track_number, track in enumerate(self.tracks):
            index, track_id = track['registry_index'], track['track_id']
            # A total describes the entire model output window, not the requested BED subinterval.
            self.totals.writerow([window.identifier, window.chrom, window.start,
                min(window.start + output_length, window.chrom_length), index, track_id,
                _number(values['total_z'][track_number]) if valid_count else 'NA', valid_count])
            if self.bins is not None:
                for i in selected:
                    self.bins.writerow([window.chrom, starts[i], ends[i], window.identifier, i, index, track_id,
                        *[_number(values[k][i, track_number]) for k in (
                            'signal_z', 'signal_registry', 'peak_logits', 'peak_probability',
                            'profile_logits', 'profile_probability')],
                        int(valid[i]), valid_bp[i], requested_bp[i]])
            if self.bedgraphs is not None:
                for kind in ('signal_registry', 'peak_probability', 'profile_probability'):
                    self.bedgraphs.write(index, kind, (
                        f'{window.chrom}\t{starts[i]}\t{ends[i]}\t{_number(values[kind][i, track_number])}\n'
                        for i in selected if kind != 'profile_probability' or valid[i]))
        return len(selected)

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()
        if self.bedgraphs is not None:
            self.bedgraphs.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _number(value):
    # Nine significant digits round-trip a float32 prediction.
    return format(float(value), '.9g')
