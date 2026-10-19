"""ASCEND preparation, fixed-budget LoRA training, and resumable generation evaluation."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from metrics import HAN, collapse_flags, normalize, summarize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "artifacts"


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def preparation_identity(config):
    return {key: config[key] for key in
            ("dataset", "dataset_revision", "model", "model_revision", "language_prompt")}


def check_preparation(split, config):
    ready = DATA / split / "ready.json"
    if not ready.exists():
        raise RuntimeError(f"Prepare incomplete: {split}; run asr.py prepare")
    if read_json(ready).get("preparation_identity") != preparation_identity(config):
        raise RuntimeError(f"Cache preparation identity mismatch: {split}; archive data/ and run asr.py prepare")


def evaluation_identity(group, split, config, limit=None):
    folder = OUT / group
    identity = {"config": config, "group": group, "split": split, "limit_per_language": limit}
    if group != "A":
        marker = folder / "training_complete.json"
        if not marker.exists():
            raise RuntimeError(f"Training incomplete: {group}")
        complete = read_json(marker)
        if read_json(folder / "config.json") != config or complete.get("config_sha256") != signature(config):
            raise RuntimeError(f"Evaluation must match the frozen training config: {group}")
        if group in ("B", "C") and complete["completed_steps"] != config["steps"]:
            raise RuntimeError(f"Full training budget not completed: {group}")
        identity["adapter_sha256"] = hashlib.sha256((folder / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    return identity


def check_predictions(predictions, references):
    expected = {row["id"]: (row["reference"], row["language"], row["duration"]) for row in references}
    seen = set()
    for row in predictions:
        index = row["id"]
        if type(index) is not int or index in seen or index not in expected:
            raise ValueError(f"Duplicate or foreign prediction ID: {index}")
        if (row["reference"], row["language"], row["duration"]) != expected[index]:
            raise ValueError(f"Prediction reference/language/duration mismatch: {index}")
        seen.add(index)
    return seen


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def processor(config):
    return WhisperProcessor.from_pretrained(config["model"], revision=config["model_revision"])


def prepare(config):
    for split in ("train", "validation", "test"):
        if (DATA / split / "ready.json").exists():
            check_preparation(split, config)
    seed_all(config["seed"])
    proc = processor(config)
    dataset = load_dataset(config["dataset"], revision=config["dataset_revision"])
    dataset = dataset.cast_column("audio", Audio(decode=False))
    DATA.mkdir(exist_ok=True)
    audit = {"dataset": config["dataset"], "revision": config["dataset_revision"], "splits": {}}
    for split in ("train", "validation", "test"):
        source = dataset[split]
        target = DATA / split
        target.mkdir(exist_ok=True)
        ready = target / "ready.json"
        if ready.exists():
            audit["splits"][split] = read_json(ready)
            continue
        temporary = target / "features.tmp.npy"
        features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                            shape=(len(source), 80, 3000))
        manifest = []
        mismatches = []
        counts, seconds = Counter(), Counter()
        for index, row in enumerate(source):
            audio = row["audio"]
            wave, rate = sf.read(io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"], dtype="float32")
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            if rate != 16000 or not 0 < len(wave) <= 480000 or not np.isfinite(wave).all():
                raise ValueError(f"Invalid 16kHz <=30s audio: {split}/{index}, rate={rate}, n={len(wave)}")
            lang = row["language"]
            if lang not in config["language_prompt"]:
                raise ValueError(f"Unexpected language: {lang}")
            text = row["transcription"]
            chinese = bool(re.search(HAN, text))
            english = bool(re.search(r"[a-zA-Z]", text))
            inferred = "mixed" if chinese and english else "zh" if chinese else "en"
            if inferred != lang:
                mismatches.append({"id": index, "language": lang, "text_inferred": inferred, "text": text})
            proc.tokenizer.set_prefix_tokens(language=config["language_prompt"][lang], task="transcribe")
            label = proc.tokenizer(text).input_ids
            if len(label) >= 448:
                raise ValueError(f"Reference exceeds Whisper label budget: {split}/{index}")
            extracted = proc.feature_extractor(wave, sampling_rate=rate, return_tensors="np", return_attention_mask=True)
            features[index] = extracted.input_features[0]
            duration = len(wave) / rate
            manifest.append({"id": index, "language": lang, "reference": text, "duration": duration,
                             "labels": label[1:], "frames": min(3000, (len(wave) + 159) // 160),
                             "source_id": str(row.get("id", audio.get("path", index)))})
            counts[lang] += 1
            seconds[lang] += duration
            if (index + 1) % 500 == 0:
                print(f"prepare {split} {index + 1}/{len(source)}", flush=True)
        features.flush()
        del features
        temporary.replace(target / "features.npy")
        save_json(target / "manifest.json", manifest)
        save_json(target / "language_mismatches.json", mismatches)
        summary = {"n": len(source), "counts": dict(counts), "hours": sum(seconds.values()) / 3600,
                   "hours_by_language": {k: v / 3600 for k, v in seconds.items()},
                   "text_language_mismatch_count": len(mismatches),
                   "max_label_tokens": max(len(row["labels"]) for row in manifest),
                   "model_revision": config["model_revision"], "dataset_revision": config["dataset_revision"],
                   "preparation_identity": preparation_identity(config)}
        save_json(ready, summary)
        audit["splits"][split] = summary
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    save_json(OUT / "data_audit.json", audit)


def load_split(split, config):
    check_preparation(split, config)
    return read_json(DATA / split / "manifest.json"), np.load(DATA / split / "features.npy", mmap_mode="r")


def sample_indices(rows, group, config, count):
    rng = np.random.default_rng(config["seed"])
    weights = np.ones(len(rows), dtype=np.float64)
    if group == "C":
        mixed = np.array([row["language"] == "mixed" for row in rows])
        probability = config["mixed_probability_C"]
        if not 0 < mixed.sum() < len(rows):
            raise ValueError("Need both mixed and monolingual training examples")
        weights[mixed] = probability / mixed.sum()
        weights[~mixed] = (1 - probability) / (~mixed).sum()
    elif group not in ("B", "pilot"):
        raise ValueError(group)
    return rng.choice(len(rows), size=count, replace=True, p=weights / weights.sum())


def base_model(config):
    return WhisperForConditionalGeneration.from_pretrained(
        config["model"], revision=config["model_revision"], attn_implementation="sdpa")


def optimizer_update(model, optimizer, scaler, batch, max_grad_norm):
    """Retry overflowed updates without changing data, randomness, or the step budget."""
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    for attempt in range(16):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        loss_sum = 0.0
        for features, label in batch:
            inputs = torch.from_numpy(features).to("cuda")
            labels = torch.tensor([label], device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_features=inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss_sum += loss.detach().item()
            scaler.scale(loss / len(batch)).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= scale_before:
            return loss_sum / len(batch), float(norm)
        print(f"FP16 overflow: retry {attempt + 1}, scale={scaler.get_scale()}", flush=True)
    raise RuntimeError("FP16 overflow persists after 16 retries")


def load_training_state(group, steps, config):
    if group not in ("pilot", "B", "C") or not 0 < steps <= config["steps"]:
        raise ValueError("Training requires pilot/B/C and 0 < steps <= the frozen budget")
    if group in ("B", "C") and steps != config["steps"]:
        raise ValueError("B/C require the full frozen budget; use pilot for shorter training")
    folder = OUT / group
    frozen = folder / "config.json"
    if frozen.exists() and read_json(frozen) != config:
        raise RuntimeError("Run config changed; use a new run directory")
    resume = folder / "resume.pt"
    marker = folder / "training_complete.json"
    if not resume.exists():
        if (folder / "adapter").exists() or marker.exists():
            raise RuntimeError("Adapter or completion marker exists without resume.pt; archive the run before retraining")
        return None
    if not frozen.exists():
        raise RuntimeError("Checkpoint lacks its frozen training config")
    state = torch.load(resume, map_location="cpu", weights_only=False)
    if not 0 <= state["step"] <= steps:
        raise RuntimeError("Requested steps are less than the saved checkpoint step")
    if marker.exists() and state["step"] != steps:
        raise RuntimeError("Completion marker conflicts with checkpoint step; archive the run before retraining")
    if state["step"] == steps:
        if not marker.exists() or read_json(marker) != {"completed_steps": steps, "config_sha256": signature(config)}:
            raise RuntimeError("Target checkpoint has missing or inconsistent completion metadata")
        counters = {int(value["step"]) for value in state["optimizer"]["state"].values()}
        if state["scheduler"]["last_epoch"] != steps or counters != {steps}:
            raise RuntimeError("Completed checkpoint optimizer/scheduler counters are inconsistent")
    return state


def train(group, steps, config):
    state = load_training_state(group, steps, config)
    if state is not None and state["step"] == steps:
        print(f"{group}: checkpoint already at {steps} steps; no files changed", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: no CPU fallback for hardware acceptance")
    seed_all(config["seed"])
    rows, features = load_split("train", config)
    folder = OUT / group
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / "resume.pt"
    save_json(folder / "config.json", config)
    count = config["steps"] * config["gradient_accumulation"]
    indices = sample_indices(rows, group, config, count)
    np.save(folder / "sample_indices.npy", indices)
    consumed = indices[:steps * config["gradient_accumulation"]]
    sample_stats = {"sample_count": len(consumed), "language_counts": dict(Counter(rows[i]["language"] for i in consumed)),
                    "actual_speech_hours": sum(rows[i]["duration"] for i in consumed) / 3600,
                    "padded_input_hours": len(consumed) * 30 / 3600,
                    "unique_examples": len(set(map(int, consumed)))}
    save_json(folder / "sampling.json", sample_stats)
    model = base_model(config)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        target_modules=config["lora_targets"], lora_dropout=config["lora_dropout"], bias="none"))
    initial = get_peft_model_state_dict(model)
    digest = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in initial.values())).hexdigest()
    save_json(folder / "initialization.json", {"adapter_sha256": digest, "seed": config["seed"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "config_sha256": signature(config), "gpu": torch.cuda.get_device_name(0)})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda").train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], config["steps"])
    scaler = torch.amp.GradScaler("cuda")
    first = state["step"] if state is not None else 0
    if state is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    elapsed_steps = []
    for step in range(first, steps):
        began = time.monotonic()
        selected = indices[step * config["gradient_accumulation"]:(step + 1) * config["gradient_accumulation"]]
        batch = [(np.array(features[i:i + 1], dtype=np.float32), rows[i]["labels"]) for i in selected]
        loss_mean, norm = optimizer_update(model, optimizer, scaler, batch, config["max_grad_norm"])
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed_steps.append(time.monotonic() - began)
        event = {"step": step + 1, "loss": loss_mean,
                 "lr": scheduler.get_last_lr()[0], "seconds": elapsed_steps[-1],
                 "grad_norm": float(norm), "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                 "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}
        with open(folder / "train.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps({"group": group, **event}), flush=True)
        if (step + 1) % 50 == 0 or step + 1 == steps:
            model.save_pretrained(folder / "adapter")
            state = {"step": step + 1, "adapter": get_peft_model_state_dict(model),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
                     "python_rng": random.getstate()}
            torch.save(state, folder / "resume.tmp.pt")
            (folder / "resume.tmp.pt").replace(resume)
            save_json(folder / "progress.json", {"completed_steps": step + 1, "target_steps": steps,
                "wall_seconds_this_session": time.monotonic() - started,
                "mean_step_seconds_this_session": float(np.mean(elapsed_steps)),
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20})
    save_json(folder / "training_complete.json", {"completed_steps": steps, "config_sha256": signature(config)})


def evaluate(group, split, config, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("limit_per_language must be positive")
    identity = evaluation_identity(group, split, config, limit)
    seed_all(config["seed"])
    rows, features = load_split(split, config)
    if limit:
        # Fixed first N per language gives the pilot coverage of all three subsets.
        selected = []
        for lang in ("mixed", "zh", "en"):
            selected.extend(row for row in rows if row["language"] == lang)
        counts = Counter()
        rows = [row for row in selected if (counts.update([row["language"]]) or counts[row["language"]] <= limit)]
    folder = OUT / group
    folder.mkdir(parents=True, exist_ok=True)
    name = split + (f"_first{limit}_per_language" if limit else "")
    output = folder / f"{name}_predictions.jsonl"
    metadata = folder / f"{name}_eval_config.json"
    if metadata.exists() and read_json(metadata) != identity:
        raise RuntimeError("Evaluation changed; refuse to mix predictions")
    if output.exists() and not metadata.exists():
        raise RuntimeError("Existing predictions lack evaluation identity; archive them before regenerating")
    predictions = []
    if output.exists():
        with open(output, encoding="utf-8") as handle:
            predictions = [json.loads(line) for line in handle]
    done = check_predictions(predictions, rows)
    save_json(metadata, identity)
    proc = processor(config)
    model = base_model(config)
    if group != "A":
        model = PeftModel.from_pretrained(model, folder / "adapter")
    model.to(device="cuda", dtype=torch.float16).eval()
    generation = copy.deepcopy(model.generation_config)
    generation.forced_decoder_ids = None
    generation.suppress_tokens = []
    generation.begin_suppress_tokens = config["begin_suppress_tokens"]
    generation.do_sample = False
    generation.num_beams = config["num_beams"]
    generation.max_new_tokens = config["max_new_tokens"]
    language_ids = set(generation.lang_to_id.values())
    started = time.monotonic()
    for lang in ("mixed", "zh", "en"):
        pending = [row for row in rows if row["language"] == lang and row["id"] not in done]
        for start in range(0, len(pending), config["eval_batch_size"]):
            batch = pending[start:start + config["eval_batch_size"]]
            inputs = torch.tensor(np.stack([features[r["id"]] for r in batch]), device="cuda", dtype=torch.float16)
            masks = torch.zeros((len(batch), 3000), dtype=torch.long, device="cuda")
            for j, row in enumerate(batch):
                masks[j, :row["frames"]] = 1
            language = config["language_prompt"][lang]
            with torch.inference_mode():
                generated = model.generate(input_features=inputs, attention_mask=masks,
                    generation_config=generation, language=language, task="transcribe", return_timestamps=False,
                    return_dict_in_generate=True).sequences
            ids = generated.cpu().tolist()
            texts = proc.batch_decode(generated, skip_special_tokens=True)
            for row, sequence, text in zip(batch, ids, texts):
                item = {"id": row["id"], "language": lang, "duration": row["duration"],
                        "reference": row["reference"], "prediction": text,
                        "normalized_prediction": normalize(text), "token_ids": sequence,
                        "collapse_flags": collapse_flags(text, sequence, generation.lang_to_id[f"<|{language}|>"],
                            language_ids, hit_token_limit=proc.tokenizer.eos_token_id not in sequence)}
                predictions.append(item)
                with open(output, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            if len(predictions) % 100 < config["eval_batch_size"]:
                print(f"evaluate {group}/{name}: {len(predictions)}/{len(rows)} ({time.monotonic()-started:.1f}s this session)", flush=True)
    expected = {row["id"] for row in rows}
    if {row["id"] for row in predictions} != expected or len(predictions) != len(expected):
        raise RuntimeError("Incomplete or duplicated predictions")
    result = summarize(predictions)
    save_json(folder / f"{name}_metrics.json", result)
    print(json.dumps({"group": group, "split": name, "metrics": result}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "eval"])
    parser.add_argument("--group", choices=["A", "B", "C", "pilot"], default="A")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--limit-per-language", type=int)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if args.action == "prepare":
        prepare(config)
    elif args.action == "train":
        train(args.group, config["steps"] if args.steps is None else args.steps, config)
    else:
        evaluate(args.group, args.split, config, args.limit_per_language)


if __name__ == "__main__":
    main()
