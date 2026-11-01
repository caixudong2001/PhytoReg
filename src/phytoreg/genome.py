"""Bounded-memory FASTA access and reference-aligned genomic windows."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import gzip
from pathlib import Path
import tempfile

import numpy as np


def open_text(path):
    path = Path(path)
    return gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz' else path.open(encoding='utf-8')


class FastaReference:
    """Normalize a FASTA into a temporary, seekable base store; never load a genome in RAM."""
    def __init__(self, path, temp_dir=None):
        self.path = Path(path)
        self.sequences = {}
        self.ambiguous_bases = 0
        self._store = tempfile.TemporaryFile(dir=temp_dir)
        try:
            opener = gzip.open if self.path.suffix == '.gz' else open
            with opener(self.path, 'rb') as handle:
                name = None
                at_line_start = True
                while True:
                    part = handle.readline(1 << 20)
                    if not part:
                        break
                    if at_line_start and part.startswith(b'>'):
                        if not part.endswith(b'\n') and len(part) == 1 << 20:
                            raise ValueError('FASTA header is too long')
                        fields = part[1:].strip().split()
                        if not fields:
                            raise ValueError('Empty FASTA sequence identifier')
                        name = fields[0].decode('utf-8')
                        if name in self.sequences:
                            raise ValueError(f'Duplicate FASTA sequence identifier: {name}')
                        self.sequences[name] = [self._store.tell(), 0]
                    else:
                        bases = part.translate(None, b' \t\r\n').upper()
                        if bases:
                            if name is None:
                                raise ValueError('FASTA sequence appears before a header')
                            if bases.translate(None, b'ACGTNRYSWKMBDHV'):
                                raise ValueError(f'Invalid DNA character in FASTA sequence {name}')
                            self.ambiguous_bases += len(bases.translate(None, b'ACGTN'))
                            bases = bases.translate(bytes.maketrans(b'RYSWKMBDHV', b'NNNNNNNNNN'))
                            self._store.write(bases)
                            self.sequences[name][1] += len(bases)
                    at_line_start = part.endswith(b'\n')
            if not self.sequences or any(length == 0 for _, length in self.sequences.values()):
                raise ValueError('FASTA must contain nonempty sequences')
            self._store.flush()
        except BaseException:
            self.close()
            raise

    @property
    def sizes(self):
        return {name: info[1] for name, info in self.sequences.items()}

    def fetch(self, chrom, start, end):
        """Fetch reference-forward bases; pad missing context with N."""
        if end < start:
            raise ValueError('FASTA fetch end precedes start')
        offset, length = self.sequences[chrom]
        left = min(end - start, max(0, -start))
        lo, hi = max(0, min(length, start)), max(0, min(length, end))
        self._store.seek(offset + lo)
        sequence = self._store.read(hi - lo)
        right = end - start - left - len(sequence)
        return b'N' * left + sequence + b'N' * right

    def close(self):
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int
    end: int
    name: str = '.'
    strand: str = '.'


def read_bed(path, sizes):
    """Read BED3-6 in zero-based half-open coordinates, validating every interval."""
    regions = []
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith('#') or text.split()[0] in ('track', 'browser'):
                continue
            fields = text.split()
            if not 3 <= len(fields) <= 6:
                raise ValueError(f'BED line {line_number}: expected 3 to 6 fields')
            chrom = fields[0]
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f'BED line {line_number}: coordinates must be integers') from exc
            if chrom not in sizes:
                raise ValueError(f'BED line {line_number}: {chrom} is absent from FASTA')
            if not 0 <= start < end <= sizes[chrom]:
                raise ValueError(f'BED line {line_number}: interval must lie within {chrom} [0,{sizes[chrom]})')
            strand = fields[5] if len(fields) == 6 else '.'
            if strand not in ('.', '+', '-'):
                raise ValueError(f'BED line {line_number}: invalid strand')
            regions.append(Region(chrom, start, end, fields[3] if len(fields) >= 4 else '.', strand))
    if not regions:
        raise ValueError('BED contains no intervals')
    return regions


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


class IntervalIndex:
    def __init__(self, regions):
        grouped = {}
        for region in regions:
            grouped.setdefault(region.chrom, []).append((region.start, region.end))
        self.intervals = {chrom: merge_intervals(values) for chrom, values in grouped.items()}
        self.ends = {chrom: [b for _, b in values] for chrom, values in self.intervals.items()}

    def overlapping(self, chrom, start, end):
        values = self.intervals.get(chrom, [])
        first = bisect_right(self.ends.get(chrom, []), start)
        for index in range(first, len(values)):
            a, b = values[index]
            if a >= end:
                break
            yield max(a, start), min(b, end)

    def coverage(self, chrom, starts, ends):
        result = np.zeros(len(starts), dtype=np.int64)
        for a, b in self.overlapping(chrom, int(starts[0]), int(ends[-1])):
            first = int(np.searchsorted(ends, a, side='right'))
            last = int(np.searchsorted(starts, b, side='left'))
            result[first:last] += np.maximum(0, np.minimum(ends[first:last], b) - np.maximum(starts[first:last], a))
        return result


@dataclass(frozen=True)
class Window:
    number: int
    chrom: str
    start: int
    chrom_length: int

    @property
    def identifier(self):
        return f'w{self.number:09d}'


def tile_ranges(requests, output_length):
    return {chrom: merge_intervals([(a // output_length, (b + output_length - 1) // output_length)
                                    for a, b in intervals])
            for chrom, intervals in requests.intervals.items()}


def plan_windows(sizes, requests, output_length):
    ranges = tile_ranges(requests, output_length)
    number = 0
    for chrom, length in sizes.items():
        for first, stop in ranges.get(chrom, []):
            for tile in range(first, stop):
                number += 1
                yield Window(number, chrom, tile * output_length, length)


def count_windows(requests, output_length):
    return sum(stop - first for spans in tile_ranges(requests, output_length).values() for first, stop in spans)


def window_arrays(window, output_length, requests, valid_regions=None):
    starts = window.start + np.arange(output_length // 32, dtype=np.int64) * 32
    ends = np.minimum(starts + 32, window.chrom_length)
    # For wholly padded bins, use zero-length intervals instead of inverted coordinates.
    ends = np.maximum(starts, ends)
    valid_bp = ends - starts
    requested_bp = requests.coverage(window.chrom, starts, ends)
    valid = valid_bp > 0
    if valid_regions is not None:
        valid &= valid_regions.coverage(window.chrom, starts, ends) == valid_bp
    return starts, ends, valid_bp, requested_bp, valid


_TOKENS = np.full(256, 4, dtype=np.int64)
_TOKENS[list(b'ACGT')] = np.arange(4)


def encode_dna(sequence):
    return _TOKENS[np.frombuffer(sequence, dtype=np.uint8)]
