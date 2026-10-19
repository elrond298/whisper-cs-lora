"""Prepare only the two pinned FLEURS test subsets; never load other splits."""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
REPO = "google/fleurs"
SR = 16000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_tsv(text, language):
    candidates, excluded = [], defaultdict(int)
    seen = set()
    for fields in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(fields) != 7:
            raise ValueError("Expected seven FLEURS TSV fields")
        sentence_id, filename, raw, reference, _, frames, gender = fields
        if filename in seen or PurePosixPath(filename).name != filename or "\\" in filename:
            raise ValueError("Unsafe or duplicate recording filename")
        seen.add(filename)
        frames = int(frames)
        if not reference.strip():
            excluded["empty_reference"] += 1
        elif frames <= 0 or frames > 30 * SR:
            excluded["duration_outside_0_30s"] += 1
        elif language == "zh" and re.search(r"[A-Za-z]", reference):
            excluded["Latin_in_Mandarin_reference"] += 1
        else:
            candidates.append({"filename": filename, "sentence_id": sentence_id,
                               "raw_reference": raw, "reference": reference,
                               "frames": frames, "gender": gender})
    return candidates, dict(excluded), len(seen)


def sample_sentences(candidates, n=300, seed=42):
    if n <= 0 or n >= len(candidates):
        raise ValueError("Need a positive strict subset")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(candidates, key=lambda r: r["filename"]):
        groups[row["sentence_id"]].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    for rows in groups.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < n:
        for key in keys:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())
    return selected


def extract_selected(archive_path, selected, destination, language, config):
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {row["filename"]: row for row in selected}
    completed = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Unsafe archive path")
            if path.name not in wanted:
                continue
            if not member.isfile() or path.name in completed:
                raise ValueError("Non-regular or duplicate selected archive member")
            if member.size > SR * 30 * 8 + 65536:
                raise ValueError("Unexpectedly large selected recording")
            row = wanted[path.name]
            with archive.extractfile(member) as handle:
                payload = handle.read()
            audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
            if rate != SR or audio.ndim != 1 or len(audio) != row["frames"] or not np.isfinite(audio).all():
                raise ValueError("Audio does not match official sample count/rate")
            target = destination / path.name
            temporary = target.with_suffix(".wav.tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
            completed[path.name] = {
                "id": f"fleurs:{config}:test:{path.name}", "source_id": path.name,
                "language": language, "audio_path": str(target.relative_to(ROOT)),
                "audio_sha256": hashlib.sha256(payload).hexdigest(),
                "reference": row["reference"], "raw_reference": row["raw_reference"],
                "duration": row["frames"] / SR, "frames": row["frames"], "sample_rate": SR,
                "cluster_id": row["sentence_id"], "sentence_id": row["sentence_id"],
                "gender": row["gender"], "dataset": REPO, "config": config, "revision": REVISION,
            }
    if set(completed) != set(wanted):
        raise ValueError(f"Missing {len(set(wanted) - set(completed))} selected audio files")
    return [completed[row["filename"]] for row in selected]


def prepare(config, language, n=300):
    folder = ROOT / "data" / "external_eval" / f"fleurs_{language}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": REPO, "revision": REVISION, "config": config, "split": "test",
                "seed": 42, "n": n, "max_seconds": 30, "no_Latin_reference": language == "zh",
                "selection": "round-robin sentence IDs, seeded shuffle", "script_sha256": sha256(__file__)}
    ready = folder / "ready.json"
    if ready.exists():
        saved = json.loads(ready.read_text())
        if saved["identity"] != identity or saved["manifest_sha256"] != sha256(folder / "manifest.json"):
            raise ValueError("Prepared subset identity changed")
        for row in json.loads((folder / "manifest.json").read_text()):
            if sha256(ROOT / row["audio_path"]) != row["audio_sha256"]:
                raise ValueError("Prepared audio changed")
        print(f"{config}: existing subset verified", flush=True)
        return
    tsv = Path(hf_hub_download(REPO, f"data/{config}/test.tsv", repo_type="dataset", revision=REVISION))
    candidates, excluded, total = parse_tsv(tsv.read_text(encoding="utf-8"), language)
    selected = sample_sentences(candidates, n)
    print(f"{config}: total={total}, eligible={len(candidates)}, selected={len(selected)}, excluded={excluded}", flush=True)
    archive = Path(hf_hub_download(REPO, f"data/{config}/audio/test.tar.gz", repo_type="dataset", revision=REVISION))
    rows = extract_selected(archive, selected, folder / "audio", language, config)
    write_json(folder / "manifest.json", rows)
    audit = {"identity": identity, "total_test_recordings": total, "eligible_recordings": len(candidates),
             "selected_recordings": len(rows), "distinct_sentences": len({r["cluster_id"] for r in rows}),
             "excluded": excluded, "selected_seconds": sum(r["duration"] for r in rows),
             "manifest_sha256": sha256(folder / "manifest.json"),
             "metadata_sha256": sha256(tsv), "archive_sha256": sha256(archive),
             "official_archive_bytes": archive.stat().st_size,
             "selected_audio_bytes": sum((ROOT / r["audio_path"]).stat().st_size for r in rows),
             "network_note": "hf_hub_download cached retrieval; archive payload size, not measured retry/network overhead",
             "reference_field": "transcription (TSV column 4); raw_transcription retained",
             "license": "FLEURS CC-BY-4.0; cite official source", "speaker_id_available": False}
    write_json(folder / "audit.json", audit)
    write_json(ready, {"identity": identity, "manifest_sha256": audit["manifest_sha256"]})
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()
    for config, language in (("cmn_hans_cn", "zh"), ("en_us", "en")):
        prepare(config, language, args.n)
