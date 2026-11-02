"""Behavioral checks for two-stage predictions, frozen first stage and bundles."""
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys

import torch
from phytoreg import PhytoReg, BackboneConfig, HeadConfig, PhytoRegConfig, SpatialConfig

ROOT = Path(__file__).resolve().parents[1]


def small_model():
    backbone = BackboneConfig(input_length_bp=1024, downstream_output_length_bp=512,
        token_embed_dim=8, conv_init_embed_dim=8, encoder_channels=(12,16,20,24,28,30,32),
        transformer_dim=32, transformer_layers=2, transformer_heads=4, transformer_ffn_dim=64,
        gradient_checkpointing=False)
    config = PhytoRegConfig(backbone, HeadConfig(8, 4, 4, 7, 2), spatial=SpatialConfig(4, 4, 3))
    torch.manual_seed(17)
    catalog = [dict(registry_index=i, head_row=row, species_id=sp, task_group=task,
                    peak_enabled=True, spatial_condition_index=condition)
        for row, (i, sp, task, condition) in enumerate([
            (1, 'a', 'Histone', 0), (2, 'a', 'OCR', 1), (4, 'b', 'TF', 2), (6, 'b', 'OCR', 1)])]
    model = PhytoReg(config, [1, 2, 4, 6], [0, 0, 1, 1], track_catalog=catalog)
    with torch.no_grad():
        for name, parameter in model.head.named_parameters():
            if name.startswith(('weights.', 'biases.', 'signal_delta')):
                parameter.normal_(0, .1)
        model.head.feature_mean.normal_(0, .1)
        model.head.feature_std.uniform_(.8, 1.2)
        model.label_center.normal_(0, .2)
        model.label_scale.uniform_(.8, 1.2)
        model.peak_decoder.spatial.readout_weight.normal_(0, .03)
        model.peak_decoder.task_readout.weight.normal_(0, .03)
        model.peak_decoder.context_gain.copy_(torch.tensor([1., 1., .75]))
        model.peak_decoder.readout_gain.copy_(torch.tensor([1., .9, .9]))
    return model


class PhytoRegTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_and_token_dna_are_equal_and_peak_uses_both_stages(self):
        model = small_model().eval()
        dna = torch.randint(0, 5, (2, 1024)); tracks = torch.tensor([[1, 2], [4, 6]])
        a = model(dna, tracks); b = model(torch.nn.functional.one_hot(dna, 5).float(), tracks)
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertEqual(a['signal_z'].shape, (2, 16, 2))
        self.assertEqual(a['total_z'].shape, (2, 2))
        sum(a[k].square().mean() for k in ['signal_z', 'peak_logits', 'profile_logits', 'total_z']).backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.head.parameters()))
        self.assertGreater(model.peak_decoder.spatial.stem.weight.grad.abs().sum().item(), 0)
        self.assertFalse(any('query' in name or 'lm_head' in name for name, _ in model.named_modules()))

    def test_masked_padding_order_and_empty_region(self):
        model = small_model().eval(); h = torch.randn(1, 7, 8); tracks = torch.tensor([[2, 1]])
        base = model.from_features(h, tracks)
        padded = model.from_features(h, torch.tensor([[1, 9999, -1, 2]]),
                                     track_valid_mask=torch.tensor([[True, False, False, True]]))
        for key, value in base.items():
            torch.testing.assert_close(padded[key][..., [3, 0]], value, rtol=2e-5 if key == 'peak_logits' else 0,
                                       atol=2e-6 if key == 'peak_logits' else 0)
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

    def test_training_updates_spatial_readout_and_preserves_first_stage_outputs(self):
        model = small_model().train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.head.training)
        source = {k:v.clone() for k,v in model.state_dict().items() if not k.startswith('peak_decoder.')}
        h = torch.randn(2, 65, 8, requires_grad=True); tracks = torch.tensor([[1, 2], [4, 6]])
        before = model.from_features(h, tracks)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        before['peak_logits'].square().mean().backward();optimizer.step()
        self.assertIsNone(h.grad)
        self.assertGreater(model.peak_decoder.task_readout.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.peak_decoder.context.parameters()))
        after = model.from_features(h, tracks)
        self.assertFalse(torch.equal(before['peak_logits'], after['peak_logits']))
        for key in ('signal_z', 'profile_logits', 'profile_score', 'total_z'):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        for key, value in source.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)

    def test_shared_pathway_receives_each_task_and_readout_is_task_specific(self):
        for track, task in [(1, 0), (2, 1), (4, 2)]:
            model = small_model().train()
            value = model.from_features(torch.randn(1, 65, 8), torch.tensor([[track]]))['peak_logits']
            value.square().mean().backward()
            self.assertGreater(model.peak_decoder.spatial.stem.weight.grad.abs().sum().item(), 0)
            grad = model.peak_decoder.spatial.readout_weight.grad
            self.assertGreater(grad[task].abs().sum().item(), 0)
            self.assertTrue((grad[torch.arange(3) != task] == 0).all())

    def test_masks_locality_reversal_and_disabled_peaks(self):
        model = small_model().eval(); h = torch.randn(1, 65, 8); ids = torch.tensor([[1, 2]])
        region = torch.ones(1, 65, dtype=torch.bool);region[:, 25:35] = False
        value = model.from_features(h, ids, position_valid_mask=region)['peak_logits']
        raw = model.head(h, ids, positions=region)['peak_logits']
        self.assertTrue(torch.equal(value[:, ~region[0]], raw[:, ~region[0]]))
        reverse = model.from_features(h.flip(1), ids, position_valid_mask=region.flip(1))['peak_logits'].flip(1)
        torch.testing.assert_close(value, reverse, atol=3e-6, rtol=3e-5)
        chunks = torch.cat([model.from_features(h, ids[:, i:i+1], position_valid_mask=region)['peak_logits'] for i in range(2)], -1)
        torch.testing.assert_close(value, chunks, atol=3e-6, rtol=3e-5)
        model.peak_decoder.enabled[1] = False
        disabled = model.from_features(h, ids, position_valid_mask=region)['peak_logits']
        self.assertTrue(torch.equal(disabled[:, :, 0], raw[:, :, 0]))

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
