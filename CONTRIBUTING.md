# Development

Use Python 3.10 or newer. Install the package in editable mode and run the behavioral tests:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Changes to the model must preserve the documented input/mask contracts and the single PhytoReg bundle format. Tests cover native and token DNA, gradients through the backbone and all four heads, padded/reordered track selection, invalid inputs, serialization integrity and the command-line interface.

Do not add experiment-specific architecture switches or alternative model loaders to the public package. Research experiments and historical comparisons belong outside this source tree.
