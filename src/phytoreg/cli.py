"""Inspect or run PhytoReg. No training is started by these commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .inference import predict_genome, predict_npz, read_track_ids, select_tracks
from .model import PhytoReg
from .prediction_output import write_tracks


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog='phytoreg', description='PhytoReg two-stage regulatory model')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'tracks', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('model', type=Path, help='PhytoReg model directory')
        if name in ('tracks', 'predict'):
            command.add_argument('--species', help='Species ID from the model catalog')
            command.add_argument('--task', action='append', choices=['Histone', 'OCR', 'TF'], help='Repeat to select multiple tasks')
        if name == 'tracks':
            command.add_argument('--output', type=Path, help='TSV file; defaults to stdout')
        if name == 'predict':
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument('--input', type=Path, help='NPZ with dna and track_indices')
            source.add_argument('--fasta', type=Path, help='Reference FASTA or FASTA.gz')
            command.add_argument('--bed', type=Path, help='BED3-6 requested regions; default: entire FASTA')
            command.add_argument('--valid-bed', type=Path, help='BED3-6 defining peak context and profile/total positions')
            selection = command.add_mutually_exclusive_group()
            selection.add_argument('--track-id', action='append', help='Track ID to predict; repeat for multiple tracks')
            selection.add_argument('--tracks-file', type=Path, help='One track ID per line')
            command.add_argument('--output', type=Path, required=True, help='New directory for FASTA, new NPZ file for NPZ input')
            command.add_argument('--device', default='cpu')
            command.add_argument('--batch-size', type=_positive, default=1)
            command.add_argument('--format', choices=['tsv', 'bedgraph', 'both'], default='tsv', help='FASTA output format')
            command.add_argument('--temp-dir', type=Path, help='Temporary FASTA storage directory')
            command.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    if getattr(args, 'output', None) is not None and args.output.exists():
        parser.error('Output already exists; choose a new path')
    if args.command == 'predict':
        if args.fasta is not None and not args.species:
            parser.error('--species is required with --fasta; use phytoreg inspect to list species')
        if args.input is not None and (args.bed or args.valid_bed or args.species or args.task
                or args.track_id or args.tracks_file or args.temp_dir or args.format != 'tsv'):
            parser.error('--bed/--valid-bed/--species/--task/track selection/--format/--temp-dir require --fasta')
    try:
        model = PhytoReg.from_pretrained(args.model, device=getattr(args, 'device', 'cpu'))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.command == 'inspect':
        print(json.dumps(dict(model='PhytoReg', step=model.metadata.get('step'),
            latest_full_validation_step=model.metadata.get('latest_full_validation_step'),
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            architecture='shared-spatial-task-readout', stages=2,
            head=asdict(model.config.head), spatial=asdict(model.config.spatial),
            species=sorted({r['species_id'] for r in model.track_catalog}),
            weights_sha256=model.metadata['loaded_weights_sha256']), ensure_ascii=False, indent=2))
        return
    try:
        if args.command == 'tracks':
            tracks = select_tracks(model, args.species, args.task)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x', encoding='utf-8', newline='') as handle:
                    write_tracks(handle, tracks)
            else:
                import sys
                write_tracks(sys.stdout, tracks)
            return
        if args.input:
            shapes = predict_npz(model, args.input, args.output, args.batch_size)
            print(json.dumps(dict(output=str(args.output), shapes=shapes)))
        else:
            track_ids = read_track_ids(args.tracks_file) if args.tracks_file else args.track_id
            result = predict_genome(model, fasta=args.fasta, output=args.output, species=args.species,
                bed=args.bed, valid_bed=args.valid_bed, tasks=args.task, track_ids=track_ids,
                batch_size=args.batch_size, output_format=args.format, temp_dir=args.temp_dir, quiet=args.quiet)
            print(json.dumps(dict(status=result['status'], output=str(args.output),
                windows=result['windows_completed'], bins=result['emitted_bins'], tracks=len(result['tracks']))))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


def inspect_main():
    import sys
    main(['inspect', *sys.argv[1:]])
