"""Inspect or run PhytoReg. No training is started by these commands."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .model import PhytoReg


def main(argv=None):
    parser = argparse.ArgumentParser(prog='phytoreg', description='PhytoReg independent regulatory model')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('inspect', 'predict'):
        command = commands.add_parser(name)
        command.add_argument('model', type=Path, help='PhytoReg model directory')
        if name == 'predict':
            command.add_argument('--input', type=Path, required=True, help='NPZ with dna and track_indices')
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.command == 'predict' and args.output.exists():
        parser.error('Output already exists; choose a new path')
    model = PhytoReg.from_pretrained(args.model, device=getattr(args, 'device', 'cpu'))
    if args.command == 'inspect':
        print(json.dumps(dict(model='PhytoReg', step=model.metadata.get('step'),
            latest_full_validation_step=model.metadata.get('latest_full_validation_step'),
            parameter_count=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            head=asdict(model.config.head),
            species=sorted({r['species_id'] for r in model.track_catalog}),
            weights_sha256=model.metadata['loaded_weights_sha256']), ensure_ascii=False, indent=2))
        return
    with np.load(args.input, allow_pickle=False) as data:
        if not {'dna', 'track_indices'} <= set(data.files):
            parser.error('Input NPZ requires dna and track_indices')
        allowed = {'dna', 'track_indices', 'track_valid_mask', 'position_valid_mask'}
        if set(data.files) - allowed:
            parser.error(f'Unknown inputs: {sorted(set(data.files) - allowed)}')
        tensors = {key: torch.from_numpy(data[key]).to(model.device) for key in data.files}
    with torch.inference_mode():
        outputs = model(**tensors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        np.savez_compressed(handle, **{k: v.cpu().numpy() for k, v in outputs.items()})
    print(json.dumps(dict(output=str(args.output), shapes={k: list(v.shape) for k, v in outputs.items()})))


def inspect_main():
    import sys
    main(['inspect', *sys.argv[1:]])
