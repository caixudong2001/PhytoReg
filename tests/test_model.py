"""Behavioral checks for independent outputs, input masks and checkpoint loading."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

import torch
from phytoreg import PhytoReg, BackboneConfig, HeadConfig, PhytoRegConfig

ROOT = Path(__file__).resolve().parents[1]


def small_model():
    backbone = BackboneConfig(input_length_bp=1024, downstream_output_length_bp=512,
        token_embed_dim=8, conv_init_embed_dim=8, encoder_channels=(12,16,20,24,28,30,32),
        transformer_dim=32, transformer_layers=2, transformer_heads=4, transformer_ffn_dim=64,
        gradient_checkpointing=False)
    config = PhytoRegConfig(backbone, HeadConfig(8, 4, 4, 7, 2))
    torch.manual_seed(17)
    model = PhytoReg(config, [1, 2, 4, 6], [0, 0, 1, 1])
    with torch.no_grad():
        for name, parameter in model.head.named_parameters():
            if name.startswith(('weights.', 'biases.', 'signal_delta')):
                parameter.normal_(0, .1)
        model.head.feature_mean.normal_(0, .1)
        model.head.feature_std.uniform_(.8, 1.2)
        model.label_center.normal_(0, .2)
        model.label_scale.uniform_(.8, 1.2)
    model.track_catalog = [dict(registry_index=i, head_row=row, species_id=sp, task_group='Histone')
                           for row, (i, sp) in enumerate([(1, 'a'), (2, 'a'), (4, 'b'), (6, 'b')])]
    return model


class PhytoRegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_and_token_dna_are_equal_and_all_heads_train(self):
        model = small_model().eval()
        dna = torch.randint(0, 5, (2, 1024)); tracks = torch.tensor([[1, 2], [4, 6]])
        a = model(dna, tracks); b = model(torch.nn.functional.one_hot(dna, 5).float(), tracks)
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(a['signal_z'].shape, (2, 16, 2))
        self.assertEqual(a['total_z'].shape, (2, 2))
        sum(a[k].square().mean() for k in ['signal_z', 'peak_logits', 'profile_logits', 'total_z']).backward()
        self.assertGreater(model.backbone.base_embedding.weight.grad.abs().sum().item(), 0)
        for name in ('signal', 'peak', 'profile', 'total'):
            self.assertGreater(model.head.weights[name].grad.abs().sum().item(), 0)
            self.assertGreater(model.head.projections[name][1].weight.grad.abs().sum().item(), 0)
        self.assertFalse(any('query' in name or 'lm_head' in name for name, _ in model.named_modules()))

    def test_masked_padding_order_and_empty_region(self):
        model = small_model().eval(); h = torch.randn(1, 7, 8); tracks = torch.tensor([[2, 1]])
        base = model.from_features(h, tracks)
        padded = model.from_features(h, torch.tensor([[1, 9999, -1, 2]]),
                                     track_valid_mask=torch.tensor([[True, False, False, True]]))
        for key, value in base.items():
            torch.testing.assert_close(padded[key][..., [3, 0]], value, rtol=0, atol=0)
            self.assertTrue((padded[key][..., 1:3] == 0).all())
        empty = model.from_features(h, tracks, position_valid_mask=torch.zeros(1, 7, dtype=torch.bool))
        self.assertTrue((empty['profile_score'] == 0).all())
        self.assertTrue(all(torch.isfinite(x).all() for x in empty.values()))
        all_invalid = model.from_features(h, torch.full((1, 2), -1))
        self.assertTrue(all((x == 0).all() for x in all_invalid.values()))

    def test_reject_unknown_tracks_mixed_species_and_bad_masks(self):
        model = small_model(); h = torch.randn(1, 7, 8)
        for tracks in (torch.tensor([[0]]), torch.tensor([[7]]), torch.tensor([[1, 4]])):
            with self.assertRaises(ValueError): model.from_features(h, tracks)
        with self.assertRaises(ValueError):
            model.from_features(h, torch.tensor([[1]]), position_valid_mask=torch.ones(1, 7))
        with self.assertRaises(ValueError): model(torch.full((1, 1024), 5), torch.tensor([[1]]))
        with self.assertRaises(ValueError): model.tracks_for('missing')
        self.assertEqual(model.tracks_for('a').tolist(), [1, 2])

    def test_save_reload_checksum_and_registry_integrity(self):
        model = small_model().eval(); model.metadata = {'step': 2100, 'latest_full_validation_step': 2000}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'model'; model.save_pretrained(directory)
            reloaded = PhytoReg.from_pretrained(directory)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, reloaded.state_dict()[key], rtol=0, atol=0)
            h = torch.randn(1, 5, 8); indices = torch.tensor([[4, 6]])
            for key, value in model.from_features(h, indices).items():
                torch.testing.assert_close(value, reloaded.from_features(h, indices)[key], rtol=0, atol=0)
            self.assertEqual(reloaded.model_name, 'PhytoReg')
            with self.assertRaises(FileExistsError): model.save_pretrained(directory)
            catalog_before = (directory / 'tracks.json').read_bytes()
            (directory / 'tracks.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'tracks.json'): PhytoReg.from_pretrained(directory)
            (directory / 'tracks.json').write_bytes(catalog_before)
            with (directory / 'model.pt').open('ab') as handle: handle.write(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum'): PhytoReg.from_pretrained(directory)

    def test_training_checkpointing_preserves_predictions(self):
        from dataclasses import replace
        model = small_model().train()
        checkpointed = small_model().train()
        checkpointed.backbone.config = replace(checkpointed.config.backbone, gradient_checkpointing=True)
        dna = torch.randint(0, 5, (1, 1024)); tracks = torch.tensor([[1, 2]])
        a = model(dna, tracks); b = checkpointed(dna, tracks)
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        sum(v.square().mean() for v in a.values()).backward()
        sum(v.square().mean() for v in b.values()).backward()
        for (name, p), (_, q) in zip(model.named_parameters(), checkpointed.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)

    def test_cli_and_only_supported_format(self):
        import numpy as np
        model = small_model().eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); bundle = path / 'model'; model.save_pretrained(bundle)
            environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='2')
            result = subprocess.run([sys.executable, '-m', 'phytoreg', 'inspect', str(bundle)],
                                    env=environment, check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(result.stdout)['model'], 'PhytoReg')
            dna = np.zeros((1, 1024), dtype=np.int64); tracks = np.array([[1,2]], dtype=np.int64)
            np.savez(path/'input.npz', dna=dna, track_indices=tracks)
            command = [sys.executable, '-m', 'phytoreg', 'predict', str(bundle),
                       '--input', str(path/'input.npz'), '--output', str(path/'output.npz')]
            subprocess.run(command, env=environment, check=True, capture_output=True)
            with np.load(path/'output.npz') as results:
                expected = model(torch.from_numpy(dna), torch.from_numpy(tracks))
                for key in expected:
                    np.testing.assert_array_equal(results[key], expected[key].detach().numpy())
            self.assertNotEqual(subprocess.run(command, env=environment, capture_output=True).returncode, 0)
            manifest = json.loads((bundle/'manifest.json').read_text())
            manifest['format'] = 'unsupported-model'
            (bundle/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'format'): PhytoReg.from_pretrained(bundle)



if __name__ == '__main__':
    unittest.main()
