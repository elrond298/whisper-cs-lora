"""Prepare the local-only CS-Dialogue test subset; never load a model.

uv run --no-project --python .venv/bin/python prepare_external_cs.py --inspect
uv run --no-project --python .venv/bin/python prepare_external_cs.py
uv run --no-project --python .venv/bin/python prepare_external_cs.py --verify

Only metadata and candidate WAVs are retained. The 19 compressed parts are fed
sequentially into ONE gzip decoder, then a streaming tar reader (no extractall).
CS audio/transcripts must not be redistributed; see the pinned source card.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import tarfile
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/external_eval/cs_dialogue"
REVISION = "a84770fcdf14072252373a0227321227eaac7dc7"
DATASET = "BAAI/CS-Dialogue"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
API = f"https://huggingface.co/api/datasets/{DATASET}"
SEED = 42
MAX_FRAMES = 30 * 16000
TAG = re.compile(r"<[^<>]+>")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
LATIN = re.compile(r"[A-Za-z]")
SOURCE_PATHS = {
    "README.md": "README.md",
    "text": "data/index/short_wav/test/text",
    "wav.scp": "data/index/short_wav/test/wav.scp",
    "Information_Index.txt": "data/index/total_information/Information_Index.txt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.chmod(0o600)
    tmp.replace(path)


def clean_reference(raw: str) -> str:
    return " ".join(TAG.sub(" ", raw).split())


def reference_exclusion(raw: str) -> str | None:
    if "**" in raw:
        return "unintelligible_or_privacy_masked_double_asterisk"
    cleaned = clean_reference(raw)
    if not cleaned:
        return "empty_reference_after_tag_removal"
    if "<" in cleaned or ">" in cleaned:
        raise ValueError("Malformed/unhandled annotation markup")
    if not (HAN.search(cleaned) and LATIN.search(cleaned)):
        return "not_mixed_after_tag_removal"
    return None


def parse_index(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError("Malformed or duplicate official index ID")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", fields[0]):
            raise ValueError("Unsafe source ID")
        result[fields[0]] = fields[1]
    return result


def parse_information(text: str) -> dict[str, dict]:
    lines = text.splitlines()
    header = "Speaker ID\tGender\tAge\tRegion\tDevice\tTopic\tWAVE ID\tTAG\tSCRIPT"
    if not lines or lines[0] != header:
        raise ValueError("Unexpected information table header")
    result = {}
    for line in lines[1:]:
        cells = line.split("\t", 8)
        if len(cells) != 9 or not cells[6].endswith(".wav"):
            raise ValueError("Malformed information row")
        source_id = cells[6][:-4]
        if source_id in result:
            raise ValueError("Duplicate WAVE ID")
        result[source_id] = dict(speaker_id=cells[0], topic=cells[5], tag=cells[7], script=cells[8])
    return result


def member_name(member: tarfile.TarInfo, seen: set[str]) -> str:
    """Validate even unwanted members; normalize only conventional leading './'."""
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    bits = name.rstrip("/").split("/")
    if (not name or name.startswith("/") or "\\" in name or "\x00" in name
            or re.match(r"^[A-Za-z]:", name) or any(x in ("", ".", "..") for x in bits)
            or bits[0] != "short_wav"):
        raise ValueError(f"Unsafe archive path: {member.name!r}")
    normalized = str(PurePosixPath(*bits))
    if not (member.isfile() or member.isdir()) or member.issparse():
        raise ValueError(f"Archive links/special/sparse entries forbidden: {name}")
    if normalized in seen:
        raise ValueError(f"Duplicate archive member: {normalized}")
    seen.add(normalized)
    return normalized


def extract_member(archive: tarfile.TarFile, member: tarfile.TarInfo, destination: Path) -> None:
    if not member.isfile() or destination.parent.is_symlink():
        raise ValueError("Only regular files into a dedicated non-symlink directory")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("Missing member contents")
    copied = 0
    with source, destination.open("xb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
            copied += len(block)
    destination.chmod(0o600)
    if copied != member.size:
        raise ValueError("Truncated archive member")


def inspect_audio(path: Path) -> dict:
    result = {"audio_sha256": sha256(path), "audio_bytes": path.stat().st_size}
    try:
        with sf.SoundFile(path) as audio:
            result.update(frames=int(audio.frames), sample_rate=int(audio.samplerate),
                          channels=int(audio.channels), duration=audio.frames / audio.samplerate)
            if audio.format not in ("WAV", "WAVEX"):
                reason = "invalid_audio_not_wav"
            elif audio.samplerate != 16000:
                reason = "invalid_audio_sample_rate_not_16000"
            elif audio.channels != 1:
                reason = "invalid_audio_not_mono"
            elif audio.frames <= 0:
                reason = "zero_frames"
            elif audio.frames > MAX_FRAMES:
                reason = "over_30_seconds"
            else:
                decoded = 0
                for block in audio.blocks(blocksize=65536, dtype="float32"):
                    if not np.isfinite(block).all():
                        raise ValueError("Nonfinite waveform")
                    decoded += len(block)
                if decoded != audio.frames:
                    raise ValueError("Truncated waveform")
                reason = None
            result["exclusion"] = reason
    except (RuntimeError, ValueError, OSError) as error:
        result["exclusion"] = "invalid_or_undecodable_audio"
        result["audio_error"] = str(error)
    return result


def balanced_selection(rows: list[dict], n: int, seed: int = SEED) -> list[dict]:
    """Randomized speaker round-robin; within a speaker, rotate available topics."""
    if not 0 <= n <= len(rows) or len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("Invalid selection size or duplicate IDs")
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r["source_id"]):
        groups[row["speaker_id"]][row.get("topic", "unknown")].append(row)
    queues = {}
    for speaker in sorted(groups):
        topics = sorted(groups[speaker])
        rng.shuffle(topics)
        for topic in topics:
            rng.shuffle(groups[speaker][topic])
        queue = []
        while topics:
            for topic in topics:
                queue.append(groups[speaker][topic].pop())
            topics = [t for t in topics if groups[speaker][t]]
        queues[speaker] = queue
    speakers = sorted(queues)
    rng.shuffle(speakers)
    selected = []
    while len(selected) < n:
        for speaker in speakers:
            if queues[speaker]:
                selected.append(queues[speaker].pop(0))
                if len(selected) == n:
                    break
        speakers = [s for s in speakers if queues[s]]
    rng.shuffle(selected)
    return selected


class Network:
    """Count received response-body bytes, including probes and failed attempts."""
    def __init__(self, directory: Path):
        self.path = directory / "network.json"
        self.records = json.loads(self.path.read_text()) if self.path.exists() else []

    def open(self, url: str, label: str, start: int = 0, end: int | None = None):
        headers = {"User-Agent": "CS-Dialogue-local-noncommercial-research/1.0", "Accept-Encoding": "identity"}
        if start or end is not None:
            headers["Range"] = f"bytes={start}-{'' if end is None else end}"
        # A unique query avoids stale cached range redirects, not access controls.
        separator = "&" if "?" in url else "?"
        request = urllib.request.Request(url + separator + "cs_request=" + str(time.time_ns()), headers=headers)
        record = {"source": label, "range_start": start, "range_end": end, "response_body_bytes": 0}
        self.records.append(record)
        try:
            response = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            record["http_status"] = error.code
            self.save()
            if error.code in (401, 403):
                raise PermissionError(f"Official source requires permission: HTTP {error.code}; no bypass attempted") from error
            raise
        record.update(http_status=response.status, content_range=response.headers.get("Content-Range"))
        if start and (response.status != 206 or not (record["content_range"] or "").startswith(f"bytes {start}-")):
            response.close()
            raise ValueError("Server did not honor the resume range")
        return response, record

    def save(self):
        atomic_json(self.path, self.records)

    def small(self, url: str, label: str, limit: int | None = None) -> bytes:
        response, record = self.open(url, label, end=limit - 1 if limit else None)
        try:
            data = response.read(limit) if limit else response.read()
            record["response_body_bytes"] = len(data)
            return data
        finally:
            response.close()
            self.save()


class SplitStream(io.RawIOBase):
    """Read byte-split pieces in order, verifying each pinned LFS SHA256."""
    def __init__(self, parts: list[dict], network: Network):
        self.parts, self.network = parts, network
        self.index = 0
        self.position = 0
        self.response = None
        self.record = None
        self.digest = hashlib.sha256()
        self.completed = []
        self.failures = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.index == len(self.parts):
            return 0
        part = self.parts[self.index]
        if self.response is None:
            self.response, self.record = self.network.open(BASE + part["path"], part["path"], self.position)
            print(f"Streaming part {self.index + 1}/19 at byte {self.position}", flush=True)
        try:
            block = self.response.read(min(len(buffer), part["size"] - self.position))
            if not block:
                raise EOFError("Premature end of archive part")
        except (OSError, EOFError) as error:
            self.record["error"] = str(error)
            self.response.close()
            self.response = None
            self.network.save()
            self.failures += 1
            if self.failures > 3:
                raise
            return self.readinto(buffer)
        self.record["response_body_bytes"] += len(block)
        self.digest.update(block)
        self.position += len(block)
        buffer[:len(block)] = block
        if self.position == part["size"]:
            actual = self.digest.hexdigest()
            self.response.close()
            self.response = None
            self.network.save()
            if actual != part["lfs"]["oid"]:
                raise ValueError(f"Pinned archive SHA256 mismatch: {part['path']}")
            self.completed.append({"path": part["path"], "bytes": self.position, "sha256": actual})
            self.index += 1
            self.position = 0
            self.digest = hashlib.sha256()
            self.failures = 0
        return len(block)

    def close(self):
        if self.response is not None:
            self.response.close()
            self.response = None
        self.network.save()
        super().close()


def sources(network: Network) -> tuple[list[dict], list[dict], dict]:
    directory = OUT / "source"
    directory.mkdir(parents=True, exist_ok=True)
    urls = {k: BASE + v for k, v in SOURCE_PATHS.items()}
    urls.update({"api.json": API, "tree.json": f"{API}/tree/{REVISION}?recursive=true&limit=1000"})
    for name, url in urls.items():
        path = directory / name
        if not path.exists():
            data = network.small(url, SOURCE_PATHS.get(name, name))
            path.write_bytes(data)
            path.chmod(0o600)
    api = json.loads((directory / "api.json").read_text())
    if api.get("gated") is not False or api.get("private") or api.get("disabled"):
        raise PermissionError("Official dataset is not publicly ungated; do not bypass")
    card = (directory / "README.md").read_text()
    if "cc-by-nc-sa-4.0" not in card.lower():
        raise ValueError("Review changed/unknown source license before proceeding")
    tree = json.loads((directory / "tree.json").read_text())
    parts = sorted([p for p in tree if re.fullmatch(r"data/short_wav/short_wav\.tar\.gz\d\d", p["path"])], key=lambda p: p["path"])
    if [p["path"] for p in parts] != [f"data/short_wav/short_wav.tar.gz{i:02d}" for i in range(19)]:
        raise ValueError("Expected exactly 19 official short-audio parts")
    if sum(p["size"] for p in parts) != 9595542742:
        raise ValueError("Pinned short-audio sizes changed")
    text = parse_index((directory / "text").read_text())
    wav = parse_index((directory / "wav.scp").read_text())
    info = parse_information((directory / "Information_Index.txt").read_text())
    if set(text) != set(wav) or not set(text) <= set(info) or len(text) != 6257:
        raise ValueError("Official test indices do not agree")
    rows = []
    mismatches = []
    for source_id in sorted(text):
        meta = info[source_id]
        if meta["tag"] != "<MIX>":
            continue
        member = wav[source_id]
        if PurePosixPath(member).name != source_id + ".wav":
            raise ValueError("Audio/index ID mismatch")
        if meta["script"] != text[source_id]:
            mismatches.append(source_id)
        rows.append(dict(source_id=source_id, speaker_id=meta["speaker_id"], topic=meta["topic"],
                         member=member, raw_reference=text[source_id], reference=clean_reference(text[source_id]),
                         exclusion=reference_exclusion(text[source_id])))
    if len(rows) != 1034 or len({r["speaker_id"] for r in rows}) != 30:
        raise ValueError("Pinned mixed pool/speaker counts changed")
    identity = {"dataset": DATASET, "revision": REVISION, "official_test_rows": len(text),
                "official_mixed_rows": len(rows), "official_test_mixed_speakers": 30,
                "public_api_gated": api["gated"], "authentication_used": False,
                "reference_authority": "data/index/short_wav/test/text (preserves official privacy masking)",
                "information_script_mismatch_ids": mismatches,
                "metadata": [{"local_path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(directory.iterdir()) if p.is_file()],
                "observed_annotation_tags": dict(Counter(t for r in rows for t in TAG.findall(r["raw_reference"]))) }
    return rows, parts, identity


def verify() -> dict:
    manifest_path = OUT / "manifest.json"
    audit = json.loads((OUT / "audit.json").read_text())
    rows = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != audit["manifest_sha256"] or len(rows) != audit["selected_count"]:
        raise ValueError("Manifest identity/count mismatch")
    if len({r["id"] for r in rows}) != len(rows) or not rows:
        raise ValueError("Duplicate IDs or empty subset")
    if sha256(OUT / "pool.json") != audit["pool_sha256"]:
        raise ValueError("Eligibility-pool identity mismatch")
    pool = json.loads((OUT / "pool.json").read_text())
    eligible = [r for r in pool if r["exclusion"] is None]
    draw = balanced_selection(eligible, min(500, len(eligible) // 2))
    if [r["source_id"] for r in draw] != [r["source_id"] for r in rows]:
        raise ValueError("Manifest differs from deterministic frozen draw")
    for row in rows:
        path = ROOT / row["audio_path"]
        if path.is_symlink() or not path.resolve().is_relative_to((OUT / "audio").resolve()):
            raise ValueError("Audio escapes dedicated directory")
        actual = inspect_audio(path)
        if actual["exclusion"] is not None or any(actual[k] != row[k] for k in ("audio_sha256", "frames", "sample_rate", "duration")):
            raise ValueError(f"Invalid audio identity: {row['id']}")
        if (row["dataset"] != DATASET or row["revision"] != REVISION or row["language"] != "mixed"
                or row["id"] != f"cs_dialogue:{REVISION}:test:mixed:{row['source_id']}"
                or not isinstance(row["speaker_id"], str) or not row["speaker_id"].strip()
                or row["cluster_id"] != row["speaker_id"] or reference_exclusion(row["raw_reference"]) is not None
                or row["reference"] != clean_reference(row["raw_reference"])):
            raise ValueError("Invalid manifest metadata")
    if {p.name for p in (OUT / "audio").iterdir()} != {Path(r["audio_path"]).name for r in rows}:
        raise ValueError("Unexpected retained audio")
    return {"selected": len(rows), "speakers": len({r["speaker_id"] for r in rows}),
            "manifest_sha256": sha256(manifest_path), "all_audio_hashes_and_full_decodes_verified": True}


def prepare(inspect_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.chmod(0o700)
    network = Network(OUT)
    rows, parts, identity = sources(network)
    probe_path = OUT / "archive_probes.json"
    if not probe_path.exists():
        probes = []
        for part in parts:
            prefix = network.small(BASE + part["path"], part["path"] + " [prefix probe]", limit=64)
            probes.append({"path": part["path"], "bytes": len(prefix), "prefix_hex": prefix.hex(),
                           "gzip_magic": prefix.startswith(b"\x1f\x8b\x08")})
        atomic_json(probe_path, probes)
    summary = dict(identity, text_eligible=sum(r["exclusion"] is None for r in rows),
                   text_exclusions=dict(Counter(r["exclusion"] for r in rows if r["exclusion"])),
                   archive_bytes=sum(p["size"] for p in parts), archives=len(parts))
    atomic_json(OUT / "preflight.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if inspect_only:
        return
    if (OUT / "ready.json").exists():
        print(json.dumps(verify(), indent=2), flush=True)
        return
    audio_dir = OUT / "audio"
    audio_dir.mkdir(exist_ok=True)
    if audio_dir.is_symlink() or any(audio_dir.iterdir()):
        raise ValueError("Nonempty candidate directory: inspect incomplete preparation before retrying")
    wanted = {r["member"]: r for r in rows if r["exclusion"] is None}
    found = set()
    seen = set()
    stream = SplitStream(parts, network)
    try:
        with io.BufferedReader(stream, buffer_size=1024 * 1024) as joined:
            with gzip.GzipFile(fileobj=joined, mode="rb") as uncompressed:
                with tarfile.open(fileobj=uncompressed, mode="r|") as archive:
                    for member in archive:
                        name = member_name(member, seen)
                        if name not in wanted:
                            continue
                        row = wanted[name]
                        path = audio_dir / (row["source_id"] + ".wav")
                        extract_member(archive, member, path)
                        row.update(inspect_audio(path))
                        found.add(name)
                        if row["exclusion"]:
                            path.unlink()
                        if len(found) % 100 == 0:
                            print(f"Validated {len(found)}/{len(wanted)} candidate WAVs", flush=True)
                # Validate the gzip trailer/CRC and all pinned compressed-part hashes.
                while uncompressed.read(1024 * 1024):
                    pass
        if len(stream.completed) != 19:
            raise ValueError("Incomplete split gzip stream")
    finally:
        stream.close()
        atomic_json(OUT / "archives_consumed.json", stream.completed)
        atomic_json(OUT / "pool.json", rows)
    for name in set(wanted) - found:
        wanted[name]["exclusion"] = "missing_official_audio_member"
    eligible = [r for r in rows if r["exclusion"] is None]
    n = min(500, len(eligible) // 2)
    if n <= 0:
        raise ValueError("No nonempty evaluation subset possible")
    selected = balanced_selection(eligible, n)
    selected_ids = {r["source_id"] for r in selected}
    for row in eligible:
        if row["source_id"] not in selected_ids:
            (audio_dir / (row["source_id"] + ".wav")).unlink()
    manifest = []
    for row in selected:
        path = audio_dir / (row["source_id"] + ".wav")
        manifest.append({"id": f"cs_dialogue:{REVISION}:test:mixed:{row['source_id']}", "language": "mixed",
                         "audio_path": str(path.relative_to(ROOT)), "dataset": DATASET, "revision": REVISION,
                         "cluster_id": row["speaker_id"], **{k: row[k] for k in (
                             "source_id", "speaker_id", "topic", "reference", "raw_reference",
                             "audio_sha256", "duration", "frames", "sample_rate")}})
    atomic_json(OUT / "pool.json", rows)
    atomic_json(OUT / "manifest.json", manifest)
    excluded = Counter(r["exclusion"] for r in rows if r["exclusion"])
    audio_bytes = sum((ROOT / r["audio_path"]).stat().st_size for r in manifest)
    audit = {"status": "complete_data_preparation_only_no_inference", "source_identity": identity,
             "protocol": {"seed": SEED, "target": "min(500, floor(eligible_mixed_pool/2))",
                          "reference_cleaning": "Remove all <...> annotation tags; collapse whitespace; retain spoken text/punctuation/case",
                          "eligibility": "Official short-wav test <MIX>; no **; cleaned reference has Han and ASCII Latin; decodable finite mono WAV; 16000 Hz; 0 < frames <= 480000; no cropping/resampling",
                          "exclusion_precedence": "**; empty text; nonmixed cleaned text; missing member; invalid format/rate/channels; zero frames; >30 s; full decode failure",
                          "selection": "Sorted IDs, Python Random(42), shuffled per-speaker/topic queues, topic round-robin within speaker, randomized speaker round-robin with shortage redistribution, shuffled final order",
                          "cluster_id": "Published speaker_id, not an invented dialogue ID", "draw_frozen_before_inference": True},
             "initial_mixed_pool": len(rows), "text_eligible_count": summary["text_eligible"],
             "audio_candidates_found": len(found), "eligible_count": len(eligible), "excluded_count": sum(excluded.values()),
             "exclusion_counts": dict(sorted(excluded.items())), "selected_count": n,
             "eligible_not_selected_count": len(eligible) - n, "manifest_sha256": sha256(OUT / "manifest.json"),
             "pool_sha256": sha256(OUT / "pool.json"), "selected_audio_bytes": audio_bytes,
             "selected_duration_seconds": sum(r["duration"] for r in manifest),
             "selected_duration_min_seconds": min(r["duration"] for r in manifest),
             "selected_duration_max_seconds": max(r["duration"] for r in manifest),
             "eligible_speaker_counts": dict(sorted(Counter(r["speaker_id"] for r in eligible).items())),
             "selected_speaker_counts": dict(sorted(Counter(r["speaker_id"] for r in manifest).items())),
             "eligible_topic_counts": dict(sorted(Counter(r["topic"] for r in eligible).items())),
             "selected_topic_counts": dict(sorted(Counter(r["topic"] for r in manifest).items())),
             "archives": stream.completed, "archive_members_checked": len(seen),
             "bandwidth": {"measured_response_body_bytes": sum(r["response_body_bytes"] for r in network.records),
                           "compressed_archive_bytes": sum(p["size"] for p in parts),
                           "measurement": "Preparation script HTTP response-body bytes including metadata/probes/retries; HTTP/TLS overhead and earlier source-investigation requests excluded",
                           "compressed_archives_retained_bytes": 0, "long_audio_requested": False,
                           "strategy": "64-byte prefix probe of all parts, then sequential concatenated streaming; all parts consumed for full mixed-pool and gzip/LFS integrity validation"},
             "license": {"label": "CC BY-NC-SA 4.0 plus source-card additional restrictions",
                         "source_card": BASE + "README.md", "access": "Publicly ungated, unauthenticated HTTP; no access forms or invented personal information",
                         "use": "Local noncommercial research only; no re-identification; honor withdrawals",
                         "distribution": "DO NOT redistribute CS audio/transcripts or include them in public delivery packages; research derivatives remain within research group absent written permission"},
             "limitations": ["No human listening or model inference performed", "Published speaker/topic only; no verified dialogue grouping", "No claim of freedom from pretraining contamination or proven ASCEND waveform nonoverlap", "Exact byte counts cover script response bodies, not transport overhead or earlier exploratory source queries"],
             "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    audit["local_storage_bytes_before_audit"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    atomic_json(OUT / "audit.json", audit)
    validation = verify()
    audit["verification"] = validation
    atomic_json(OUT / "audit.json", audit)
    atomic_json(ROOT / "reports/CS_EXTERNAL_DATA_AUDIT.json", audit)
    report = f"""# CS-Dialogue external subset data audit

Data preparation only: **{n} frozen mixed-test utterances**, {len(audit['selected_speaker_counts'])}/30 official test speakers. No inference, model loading, GPU use, or changes to original ASCEND artifacts.

- Source: `{DATASET}`, revision `{REVISION}`; official short-wav test indices only.
- Initial official `<MIX>` pool: **{len(rows)}**. Text-eligible: **{summary['text_eligible']}**. Final audio/text-eligible: **{len(eligible)}**. Excluded: **{sum(excluded.values())}**.
- Exclusive first-failure exclusion counts: `{json.dumps(dict(excluded), ensure_ascii=False)}`. Full per-source decisions: local `data/external_eval/cs_dialogue/pool.json`.
- Fixed sample size: `min(500, floor({len(eligible)}/2)) = {n}`, seed 42. Speaker round-robin covers all available speakers before extra allocations, redistributes shortages, and rotates available topics within each speaker. Final order randomized. No result-based selection.
- Selected speaker allocations: `{json.dumps(audit['selected_speaker_counts'], ensure_ascii=False, sort_keys=True)}`.
- Selected topic counts: `{json.dumps(audit['selected_topic_counts'], ensure_ascii=False, sort_keys=True)}`. Topic is published metadata, not inferred dialogue identity.
- Audio: 16 kHz mono, finite full decodes, nonzero, <=30 seconds; no resampling/cropping/re-encoding. Selected duration {audit['selected_duration_seconds']:.6f} s; range {audit['selected_duration_min_seconds']:.6f}–{audit['selected_duration_max_seconds']:.6f} s.
- Cleaning: remove `<...>` tags, collapse whitespace, retain words/punctuation/case; reject any `**`, then require Han + ASCII Latin in cleaned text. Official test `text` is authoritative; two information-table discrepancies contain less privacy masking and are not used to restore names.
- Manifest: `data/external_eval/cs_dialogue/manifest.json`, SHA256 `{audit['manifest_sha256']}`. Unique namespaced IDs; speaker-based cluster IDs. Every selected audio SHA256 and complete decode reverified before readiness.
- Acquisition: 19 byte-split parts read in order into one gzip stream, not treated as independent shards. Checked {len(seen)} tar member paths/types/duplicates, extracted only text-eligible mixed candidates. No extractall, traversal, links or special files. Every compressed part matches pinned LFS SHA256; full gzip trailer/CRC validated. Original remote audio untouched.
- Archive payload: {audit['bandwidth']['compressed_archive_bytes']:,} bytes; script measured HTTP response bodies: {audit['bandwidth']['measured_response_body_bytes']:,} bytes (metadata, probes, retries included). HTTP/TLS overhead and earlier source-investigation traffic excluded. No long audio or compressed archive files retained. Selected WAVs: {audio_bytes:,} bytes. Local data storage before audit: {audit['local_storage_bytes_before_audit']:,} bytes, excluding this audit/ready marker. Metadata includes the full official information table and remains local.
- Access: API `gated:false`; public unauthenticated retrieval worked without forms or identities. Source card is retained locally. **Local noncommercial research only; no redistribution of CS audio/transcripts/derivatives to third parties without required written permission.** These files must not enter public delivery packages.
- Limits: no listening/ASR inference, no verified dialogue IDs, no proven ASCEND waveform nonoverlap, and no pretraining-contamination guarantee. Parent task owns preservation checks and downstream evaluation/statistics.

Machine-readable counts, source hashes, exclusions, speaker/topic distributions, bandwidth/storage, and validation: `reports/CS_EXTERNAL_DATA_AUDIT.json` and local `audit.json`. `ready.json` is written atomically only after successful complete validation and report output.
"""
    (ROOT / "reports/CS_EXTERNAL_DATA_AUDIT.md").write_text(report, encoding="utf8")
    atomic_json(OUT / "ready.json", {"dataset": DATASET, "revision": REVISION,
                                     "manifest_sha256": audit["manifest_sha256"], "audit_sha256": sha256(OUT / "audit.json"),
                                     "selected_count": n, "prepared_at_utc": audit["prepared_at_utc"]})
    print(json.dumps(validation, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inspect", action="store_true", help="Fetch metadata and 64-byte part prefixes only")
    group.add_argument("--verify", action="store_true", help="Offline manifest, deterministic draw and full audio verification")
    args = parser.parse_args()
    if args.verify:
        result = verify()
        marker = json.loads((OUT / "ready.json").read_text())
        if marker["manifest_sha256"] != result["manifest_sha256"] or marker["audit_sha256"] != sha256(OUT / "audit.json"):
            raise ValueError("Ready marker identity mismatch")
        print(json.dumps(result, indent=2))
    else:
        prepare(args.inspect)


if __name__ == "__main__":
    main()
