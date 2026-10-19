"""Offline, self-contained CS preparation tests; no models or remote data."""
from collections import Counter
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import prepare_external_cs as cs


class ReferenceTests(unittest.TestCase):
    def test_tags_do_not_fabricate_english(self):
        self.assertEqual(cs.clean_reference(" <FIL/> 我喜欢 <SPK/> English <NON/> <NPS/> "), "我喜欢 English")
        self.assertIsNone(cs.reference_exclusion("我喜欢 English<FIL/>"))
        self.assertEqual(cs.reference_exclusion("只有中文<FIL/><SPK/>"), "not_mixed_after_tag_removal")
        self.assertEqual(cs.reference_exclusion("中文 ** English"), "unintelligible_or_privacy_masked_double_asterisk")
        self.assertEqual(cs.reference_exclusion("<FIL/>"), "empty_reference_after_tag_removal")
        with self.assertRaises(ValueError):
            cs.reference_exclusion("中文 English <broken")

    def test_official_parsers(self):
        self.assertEqual(cs.parse_index("u_1 中文 English\nu_2  another line"), {"u_1": "中文 English", "u_2": "another line"})
        for text in ("u one\nu two", "../u text", "only_id"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                cs.parse_index(text)
        header = "Speaker ID\tGender\tAge\tRegion\tDevice\tTopic\tWAVE ID\tTAG\tSCRIPT\n"
        row = "S01\tF\t20\tregion\tdevice\teducation\tu.wav\t<MIX>\t中文 English"
        parsed = cs.parse_information(header + row)
        self.assertEqual(parsed["u"], dict(speaker_id="S01", topic="education", tag="<MIX>", script="中文 English"))
        with self.assertRaises(ValueError):
            cs.parse_information(header + row + "\n" + row)


class ArchiveTests(unittest.TestCase):
    def test_reject_traversal_links_special_and_duplicates(self):
        for name in ("/short_wav/a.wav", "../short_wav/a.wav", "short_wav/../a.wav", "short_wav/x/./a.wav", "short_wav//a.wav", "C:/short_wav/a.wav", "short_wav\\a.wav", "elsewhere/a.wav"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cs.member_name(tarfile.TarInfo(name), set())
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
            member = tarfile.TarInfo("short_wav/a.wav")
            member.type = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                cs.member_name(member, set())
        seen = set()
        self.assertEqual(cs.member_name(tarfile.TarInfo("./short_wav/a.wav"), seen), "short_wav/a.wav")
        with self.assertRaises(ValueError):
            cs.member_name(tarfile.TarInfo("short_wav/a.wav"), seen)

    def test_regular_extraction_is_exact_and_never_overwrites(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            member = tarfile.TarInfo("short_wav/a.wav")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"abc"))
        with tempfile.TemporaryDirectory() as directory, tarfile.open(fileobj=io.BytesIO(buffer.getvalue())) as archive:
            member = archive.next()
            cs.member_name(member, set())
            path = Path(directory) / "a.wav"
            cs.extract_member(archive, member, path)
            self.assertEqual(path.read_bytes(), b"abc")
            with self.assertRaises(FileExistsError):
                cs.extract_member(archive, member, path)
            symlink = Path(directory) / "symlink.wav"
            symlink.symlink_to(path)
            with self.assertRaises(FileExistsError):
                cs.extract_member(archive, member, symlink)
            self.assertEqual(path.read_bytes(), b"abc")

    def test_split_parts_form_one_gzip_and_are_hashed(self):
        compressed = gzip.compress(bytes(range(256)) * 100)
        pieces = [compressed[:57], compressed[57:203], compressed[203:]]
        parts = [dict(path=str(i), size=len(p), lfs={"oid": hashlib.sha256(p).hexdigest()}) for i, p in enumerate(pieces)]

        class Network:
            def open(self, url, label, start=0):
                return io.BytesIO(pieces[int(label)][start:]), {"response_body_bytes": 0}

            def save(self):
                pass

        stream = cs.SplitStream(parts, Network())
        with redirect_stdout(io.StringIO()), io.BufferedReader(stream) as joined, gzip.GzipFile(fileobj=joined) as gz:
            self.assertEqual(gz.read(), bytes(range(256)) * 100)
        self.assertEqual(len(stream.completed), 3)
        bad = [dict(parts[0], lfs={"oid": "0" * 64}), *parts[1:]]
        with redirect_stdout(io.StringIO()), cs.SplitStream(bad, Network()) as stream, self.assertRaises(ValueError):
            stream.read()


class SelectionAndAudioTests(unittest.TestCase):
    def rows(self, sizes):
        return [dict(source_id=f"s{s}_u{i}", speaker_id=f"s{s}", topic=f"t{i % 3}")
                for s, size in enumerate(sizes) for i in range(size)]

    def test_deterministic_balanced_selection_and_shortage_redistribution(self):
        rows = self.rows([1, 30, 30, 30])
        selected = cs.balanced_selection(rows, 22)
        self.assertEqual(selected, cs.balanced_selection(list(reversed(rows)), 22))
        self.assertNotEqual(selected, cs.balanced_selection(rows, 22, seed=43))
        self.assertEqual(Counter(r["speaker_id"] for r in selected), {"s0": 1, "s1": 7, "s2": 7, "s3": 7})
        self.assertEqual(len({r["source_id"] for r in selected}), 22)
        self.assertEqual(len({r["speaker_id"] for r in cs.balanced_selection(rows, 4)}), 4)
        self.assertEqual(cs.balanced_selection(rows, 0), [])
        with self.assertRaises(ValueError):
            cs.balanced_selection(rows + [rows[0]], 1)

    def test_topic_round_robin(self):
        selected = cs.balanced_selection(self.rows([30]), 9)
        self.assertEqual(Counter(r["topic"] for r in selected), {"t0": 3, "t1": 3, "t2": 3})

    def test_duration_cap_and_full_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            for frames, expected in ((0, "zero_frames"), (1, None), (480000, None), (480001, "over_30_seconds")):
                sf.write(path, np.zeros(frames, dtype=np.float32), 16000, subtype="PCM_16")
                result = cs.inspect_audio(path)
                self.assertEqual(result["exclusion"], expected)
                self.assertEqual(result["frames"], frames)
                self.assertEqual(result["duration"], frames / 16000)
            sf.write(path, np.zeros(8000, dtype=np.float32), 8000)
            self.assertEqual(cs.inspect_audio(path)["exclusion"], "invalid_audio_sample_rate_not_16000")
            sf.write(path, np.zeros((16, 2), dtype=np.float32), 16000)
            self.assertEqual(cs.inspect_audio(path)["exclusion"], "invalid_audio_not_mono")
            path.write_bytes(b"not a wav")
            self.assertEqual(cs.inspect_audio(path)["exclusion"], "invalid_or_undecodable_audio")
            sf.write(path, np.array([np.nan], dtype=np.float32), 16000, subtype="FLOAT")
            self.assertEqual(cs.inspect_audio(path)["exclusion"], "invalid_or_undecodable_audio")


class PreparationIntegrationTests(unittest.TestCase):
    def test_pipeline_frozen_manifest_ready_and_no_ready_after_unsafe_member(self):
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "reports").mkdir()
                output = root / "data/external_eval/cs_dialogue"
                wav = io.BytesIO()
                sf.write(wav, np.zeros(160, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
                rows = [dict(source_id=f"u{i}", speaker_id=f"s{i % 2}", topic="test", member=f"short_wav/u{i}.wav",
                             raw_reference="中文 test<FIL/>", reference="中文 test", exclusion=None) for i in range(6)]
                tar = io.BytesIO()
                with tarfile.open(fileobj=tar, mode="w") as archive:
                    for row in rows:
                        member = tarfile.TarInfo(row["member"])
                        member.size = len(wav.getvalue())
                        archive.addfile(member, io.BytesIO(wav.getvalue()))
                    if unsafe:
                        archive.addfile(tarfile.TarInfo("short_wav/../outside"))
                compressed = gzip.compress(tar.getvalue())
                step = len(compressed) // 19
                pieces = [compressed[i * step:(i + 1) * step] for i in range(18)] + [compressed[18 * step:]]
                parts = [dict(path=str(i), size=len(p), lfs={"oid": hashlib.sha256(p).hexdigest()}) for i, p in enumerate(pieces)]

                class Network:
                    def __init__(self, directory):
                        self.records = []

                    def small(self, url, label, limit=None):
                        return pieces[int(url.split(cs.BASE)[1])][:limit]

                    def open(self, url, label, start=0):
                        record = {"response_body_bytes": 0}
                        self.records.append(record)
                        return io.BytesIO(pieces[int(label)][start:]), record

                    def save(self):
                        pass

                with patch.multiple(cs, ROOT=root, OUT=output, Network=Network, sources=lambda _: (rows, parts, {})), redirect_stdout(io.StringIO()):
                    if unsafe:
                        with self.assertRaises(ValueError):
                            cs.prepare()
                        self.assertFalse((output / "ready.json").exists())
                    else:
                        cs.prepare()
                        result = cs.verify()
                        marker = json.loads((output / "ready.json").read_text())
                        manifest = json.loads((output / "manifest.json").read_text())
                        self.assertEqual(result["selected"], 3)
                        self.assertEqual(marker["manifest_sha256"], cs.sha256(output / "manifest.json"))
                        self.assertEqual(marker["audit_sha256"], cs.sha256(output / "audit.json"))
                        self.assertEqual(len(list((output / "audio").iterdir())), 3)
                        self.assertTrue(all(r["cluster_id"] == r["speaker_id"] for r in manifest))
                        self.assertEqual((root / "reports/CS_EXTERNAL_DATA_AUDIT.json").read_bytes(), (output / "audit.json").read_bytes())
                        pool_path = output / "pool.json"
                        pool_path.write_text(pool_path.read_text() + "\n")
                        with self.assertRaisesRegex(ValueError, "Eligibility-pool identity"):
                            cs.verify()


if __name__ == "__main__":
    unittest.main()
