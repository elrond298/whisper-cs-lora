import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import prepare_external_fleurs as data


class FleursDataTests(unittest.TestCase):
    def test_tsv_filters_and_unique_recordings(self):
        text = '1\ta.wav\t你好\t你好\tx\t160\t0\n2\tb.wav\tJoe你好\tjoe你好\tx\t160\t1\n3\tc.wav\t长\t长\tx\t480001\t0\n'
        rows, excluded, total = data.parse_tsv(text, 'zh')
        self.assertEqual((len(rows), total), (1, 3))
        self.assertEqual(excluded, {'Latin_in_Mandarin_reference': 1, 'duration_outside_0_30s': 1})
        with self.assertRaises(ValueError):
            data.parse_tsv(text + text, 'zh')
        with self.assertRaises(ValueError):
            data.parse_tsv('1\t../a.wav\tx\tx\tx\t160\t0', 'en')

    def test_sampling_spreads_sentences_and_is_seeded(self):
        rows = [{'filename': f'{i}_{j}.wav', 'sentence_id': str(i)} for i in range(4) for j in range(3)]
        chosen = data.sample_sentences(rows, 6)
        self.assertEqual(chosen, data.sample_sentences(rows, 6))
        self.assertEqual(len({r['sentence_id'] for r in chosen}), 4)
        self.assertEqual(len({r['filename'] for r in chosen}), 6)
        with self.assertRaises(ValueError):
            data.sample_sentences(rows, 12)

    def test_safe_extraction_checks_audio_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(160), 16000, format='WAV', subtype='PCM_16')
            payload = buffer.getvalue()
            archive = root / 'test.tar.gz'
            with tarfile.open(archive, 'w:gz') as out:
                member = tarfile.TarInfo('test/a.wav')
                member.size = len(payload)
                out.addfile(member, io.BytesIO(payload))
            row = {'filename': 'a.wav', 'frames': 160, 'sentence_id': '1', 'reference': 'hello', 'raw_reference': 'Hello', 'gender': '0'}
            with patch.object(data, 'ROOT', root):
                rows = data.extract_selected(archive, [row], root / 'audio', 'en', 'en_us')
                self.assertEqual(rows[0]['duration'], 0.01)
                self.assertEqual(rows[0]['cluster_id'], '1')
                row['frames'] = 161
                with self.assertRaises(ValueError):
                    data.extract_selected(archive, [row], root / 'bad', 'en', 'en_us')
            with tarfile.open(archive, 'w:gz') as out:
                member = tarfile.TarInfo('../a.wav')
                member.size = len(payload)
                out.addfile(member, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                data.extract_selected(archive, [row], root / 'unsafe', 'en', 'en_us')


if __name__ == '__main__':
    unittest.main()
