"""Coordinate, mask, batching and complete CLI regression tests."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from phytoreg.genome import (FastaReference, IntervalIndex, Region, count_windows, encode_dna,
                            plan_windows, read_bed, window_arrays)
from phytoreg.inference import predict_genome, predict_npz, select_tracks
from phytoreg.prediction_output import BedGraphFiles
from test_model import small_model

ROOT = Path(__file__).resolve().parents[1]


def model_fixture():
    model = small_model().eval()
    for row in model.track_catalog:
        row['track_id'] = f"{row['species_id']}_{row['registry_index']}"
    return model


def table(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


class GenomeTests(unittest.TestCase):
    def test_fasta_normalization_gzip_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ref.fa.gz'
            with gzip.open(path, 'wb') as handle:
                handle.write(b'>chrA description\r\nacgt ry\r\nNN\n>chrB\nTaC\n')
            with FastaReference(path) as reference:
                self.assertEqual(reference.sizes, {'chrA': 8, 'chrB': 3})
                self.assertEqual(reference.ambiguous_bases, 2)
                self.assertEqual(reference.fetch('chrA', -2, 10), b'NNACGTNNNNNN')
                self.assertEqual(reference.fetch('chrB', 0, 3), b'TAC')
                self.assertEqual(reference.fetch('chrB', -10, -5), b'NNNNN')
                self.assertEqual(reference.fetch('chrB', 7, 10), b'NNN')
            np.testing.assert_array_equal(encode_dna(b'ACGTN'), [0, 1, 2, 3, 4])

    def test_fasta_long_unwrapped_and_reject_invalid_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ref.fa'
            path.write_bytes(b'>long\n' + b'A' * ((1 << 20) + 7) + b'CG\n>second\nT\n')
            with FastaReference(path) as reference:
                self.assertEqual(reference.sizes['long'], (1 << 20) + 9)
                self.assertEqual(reference.fetch('long', (1 << 20) + 7, (1 << 20) + 9), b'CG')
            for content in (b'', b'ACGT\n', b'>x\n>y\nAC\n', b'>x\nAC\n>x\nGT', b'>x\nAC-X'):
                path.write_bytes(content)
                with self.assertRaises(ValueError): FastaReference(path)

    def test_bed_validation_and_strand_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'regions.bed'
            path.write_text('track name=test\n# comment\nchrA\t1\t32\tfeature\t0\t-\n')
            rows = read_bed(path, {'chrA': 40})
            self.assertEqual(rows, [Region('chrA', 1, 32, 'feature', '-')])
            for content in ('chrA -1 32', 'chrA 32 32', 'chrA 0 41', 'chrZ 0 2',
                            'chrA one 2', 'chrA 0 2 n 0 ?','chrA 0 2 n 0 + extra',''):
                path.write_text(content)
                with self.assertRaises(ValueError): read_bed(path, {'chrA': 40})

    def test_aligned_unique_windows_and_partial_bins(self):
        regions = [Region('chrA', 33, 70), Region('chrA', 64, 530), Region('chrA', 1000, 1100)]
        requests = IntervalIndex(regions)
        windows = list(plan_windows({'chrA': 1100}, requests, 512))
        self.assertEqual(count_windows(requests, 512), 3)
        self.assertEqual([w.start for w in windows], [0, 512, 1024])
        actual = {}
        for window in windows:
            starts, ends, bp, requested, valid = window_arrays(window, 512, requests)
            for i in np.flatnonzero(requested):
                self.assertNotIn(int(starts[i]), actual)
                actual[int(starts[i])] = (int(ends[i]), int(requested[i]), bool(valid[i]))
        self.assertEqual(len(actual), 20)
        self.assertEqual(actual[32], (64, 31, True))
        self.assertEqual(actual[512], (544, 18, True))
        self.assertEqual(actual[992], (1024, 24, True))
        self.assertEqual(actual[1088], (1100, 12, True))

    def test_bedgraph_descriptor_limit_and_gzip_members(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'bedgraph'
            files = BedGraphFiles(directory, max_open=2)
            for offset in (0, 32):
                for track in range(5):
                    files.write(track, 'signal_registry', [f'chrA\t{offset}\t{offset+32}\t1\n'])
                    self.assertLessEqual(len(files.open), 2)
            files.close()
            for path in directory.glob('*.gz'):
                with gzip.open(path, 'rt') as handle:
                    self.assertEqual(handle.read(), 'chrA\t0\t32\t1\nchrA\t32\t64\t1\n')


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def _inputs(self, directory):
        fasta = directory / 'ref.fa'
        fasta.write_text('>chrA\n' + 'ACGT' * 275 + '\n>chrB\n' + 'T' * 33 + '\n')
        bed = directory / 'regions.bed'
        bed.write_text('chrA\t33\t70\tx\t0\t-\nchrA\t64\t530\ty\t0\t+\nchrA\t1000\t1100\tz\t0\t.\n')
        return fasta, bed

    def test_streamed_coordinates_batch_invariance_and_direct_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); fasta, bed = self._inputs(directory); model = model_fixture()
            first, second = directory/'one', directory/'two'
            result = predict_genome(model, fasta=fasta, bed=bed, output=first, species='a', batch_size=1, quiet=True, output_format='both')
            predict_genome(model, fasta=fasta, bed=bed, output=second, species='a', batch_size=2, quiet=True)
            self.assertEqual((result['windows_completed'], result['emitted_bins']), (3, 20))
            a, b = table(first/'predictions.tsv.gz'), table(second/'predictions.tsv.gz')
            self.assertEqual(len(a), 40)
            numbers = ['signal_z','signal_registry','peak_logit','peak_probability','profile_logit','profile_probability']
            for left, right in zip(a,b):
                for key in ('chrom','start','end','registry_index','requested_bp','profile_valid'):
                    self.assertEqual(left[key], right[key])
                np.testing.assert_allclose([float(left[k]) for k in numbers], [float(right[k]) for k in numbers], rtol=2e-5, atol=2e-6)
            regions = table(first/'regions.tsv'); self.assertEqual(regions[0]['strand'], '-')
            windows = table(first/'windows.tsv')
            self.assertEqual(windows[0]['input_start'], '-256')
            self.assertEqual(windows[-1]['output_end'], '1100')
            with FastaReference(fasta) as reference:
                dna = torch.from_numpy(encode_dna(reference.fetch('chrA', -256, 768)))[None]
            with torch.inference_mode():
                expected = model(dna, torch.tensor([[1,2]]))
            row = next(row for row in a if row['start']=='32' and row['registry_index']=='1')
            self.assertAlmostEqual(float(row['signal_z']), expected['signal_z'][0,1,0].item(), places=6)
            with gzip.open(first/'bedgraph/track_00001.peak_probability.bedGraph.gz','rt') as handle:
                graph = [line.split() for line in handle]
            self.assertEqual(len(graph),20)
            self.assertEqual(len({(r[0],r[1],r[2]) for r in graph}),20)
            self.assertEqual(graph[-1][:3],['chrA','1088','1100'])

    def test_valid_mask_normalizes_full_window_and_total_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);fasta,bed=self._inputs(directory);model=model_fixture()
            bed.write_text('chrA 64 70\n')
            valid=directory/'valid.bed';valid.write_text('chrA 0 32\nchrA 64 96\n')
            out=directory/'output'
            predict_genome(model,fasta=fasta,bed=bed,valid_bed=valid,output=out,species='a',quiet=True)
            rows=table(out/'predictions.tsv.gz');totals=table(out/'totals.tsv.gz')
            self.assertTrue(all(int(row['start'])==64 and int(row['requested_bp'])==6 for row in rows))
            self.assertTrue(all(0 < float(row['profile_probability']) < 1 for row in rows))
            self.assertTrue(all(row['valid_bins']=='2' and row['start']=='0' and row['end']=='512' for row in totals))
            with FastaReference(fasta) as reference:
                dna=torch.from_numpy(encode_dna(reference.fetch('chrA',-256,768)))[None]
            mask=torch.zeros(1,16,dtype=torch.bool);mask[:,[0,2]]=True
            with torch.inference_mode():
                pred=model(dna,torch.tensor([[1,2]]),position_valid_mask=mask)
            expected=pred['profile_score']/pred['profile_score'].sum(1,keepdim=True)
            self.assertAlmostEqual(float(rows[0]['profile_probability']),float(expected[0,2,0]),places=6)
            self.assertAlmostEqual(float(totals[0]['total_z']),float(pred['total_z'][0,0]),places=6)

    def test_empty_valid_window_and_full_reference_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);fasta,bed=self._inputs(directory);model=model_fixture()
            bed.write_text('chrA 0 33\n');valid=directory/'valid.bed';valid.write_text('chrB 0 33\n')
            out=directory/'masked'
            predict_genome(model,fasta=fasta,bed=bed,valid_bed=valid,output=out,species='a',quiet=True)
            self.assertTrue(all(r['total_z']=='NA' for r in table(out/'totals.tsv.gz')))
            self.assertTrue(all(float(r['profile_probability'])==0 and r['profile_valid']=='0' for r in table(out/'predictions.tsv.gz')))
            out=directory/'all'
            result=predict_genome(model,fasta=fasta,output=out,species='a',quiet=True)
            self.assertEqual(result['windows_completed'],4)
            tail=[r for r in table(out/'predictions.tsv.gz') if r['chrom']=='chrB' and r['start']=='32']
            self.assertEqual(len(tail),2)
            self.assertTrue(all(r['end']=='33' and r['valid_bp']=='1' for r in tail))

    def test_invalid_requests_no_overwrite_and_failed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);fasta,bed=self._inputs(directory);model=model_fixture()
            bad=directory/'bad.bed';bad.write_text('chrA 0 2000\n')
            with self.assertRaises(ValueError):
                predict_genome(model,fasta=fasta,bed=bad,output=directory/'bad',species='a',quiet=True)
            self.assertFalse((directory/'bad').exists())
            out=directory/'out';out.mkdir();(out/'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError): predict_genome(model,fasta=fasta,output=out,species='a',quiet=True)
            self.assertEqual((out/'keep.txt').read_text(),'keep')
            original=model.forward
            def broken(*args,**kwargs):
                result=original(*args,**kwargs);result['signal_z'][0,0,0]=float('nan');return result
            with patch.object(model,'forward',side_effect=broken):
                with self.assertRaisesRegex(ValueError,'non-finite'):
                    predict_genome(model,fasta=fasta,output=directory/'failed',species='a',quiet=True)
            self.assertEqual(json.loads((directory/'failed/run.json').read_text())['status'],'FAILED')

    def test_npz_batches_and_track_selection(self):
        model=model_fixture()
        self.assertEqual([r['registry_index'] for r in select_tracks(model,'a',track_ids=['a_2','a_1'])],[2,1])
        for ids in (['a_1','a_1'],['b_4'],['unknown']):
            with self.assertRaises(ValueError): select_tracks(model,'a',track_ids=ids)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);dna=np.zeros((3,1024),dtype=np.int64);tracks=np.tile([[1,2]],(3,1))
            np.savez(directory/'input.npz',dna=dna,track_indices=tracks)
            predict_npz(model,directory/'input.npz',directory/'one.npz',1)
            predict_npz(model,directory/'input.npz',directory/'two.npz',2)
            with np.load(directory/'one.npz') as a,np.load(directory/'two.npz') as b:
                for key in a.files:np.testing.assert_allclose(a[key],b[key],rtol=2e-5,atol=2e-6)

    def test_fasta_cli_and_track_catalog(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);fasta,bed=self._inputs(directory);bundle=directory/'model';model_fixture().save_pretrained(bundle)
            env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2')
            base=[sys.executable,'-m','phytoreg']
            proc=subprocess.run(base+['tracks',str(bundle),'--species','a'],env=env,check=True,text=True,capture_output=True)
            self.assertIn('a_1',proc.stdout);self.assertNotIn('b_4',proc.stdout)
            ids=directory/'tracks.txt';ids.write_text('a_2\n')
            out=directory/'prediction'
            proc=subprocess.run(base+['predict',str(bundle),'--fasta',str(fasta),'--bed',str(bed),'--species','a',
                '--tracks-file',str(ids),'--output',str(out),'--batch-size','2','--format','both','--quiet'],env=env,check=True,text=True,capture_output=True)
            self.assertEqual(json.loads(proc.stdout)['status'],'COMPLETED')
            self.assertTrue(all(r['track_id']=='a_2' for r in table(out/'predictions.tsv.gz')))
            proc=subprocess.run(base+['predict',str(bundle),'--fasta',str(fasta),'--output',str(directory/'missing')],env=env,capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0);self.assertIn('--species is required',proc.stderr)


if __name__ == '__main__':
    unittest.main()
