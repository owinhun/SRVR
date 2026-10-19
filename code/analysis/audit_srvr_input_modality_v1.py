#!/usr/bin/env python3
"""Audit the actual modality contract of the frozen SRVR evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
O = ROOT / "outputs"
OUT = O / "srvr_input_modality_audit_v1.json"
REPORT = O / "srvr_input_modality_audit_v1.md"
PRIMARY_PREPARER = ROOT / "src/analysis/prepare_video_salmonn2_cmtb291_v1.py"
PRIMARY_RUNNER = ROOT / "src/qwen/run_video_salmonn2_config.py"
PRIMARY_LOADER = (
    ROOT / "src/qwen/third_party/video_salmonn2/llava/dataset/av_dataset.py"
)
PRIMARY_TRAINER = (
    ROOT / "src/qwen/third_party/video_salmonn2/llava/train/llava_trainer.py"
)
PRIMARY_MODEL = (
    ROOT
    / "src/qwen/third_party/video_salmonn2/llava/model/language_model/video_salmonn_2.py"
)
SECONDARY_RUNNER = ROOT / "src/qwen/run_videollama21_av_eligibility_v1.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    preparer_source = PRIMARY_PREPARER.read_text()
    primary_loader_source = PRIMARY_LOADER.read_text()
    primary_trainer_source = PRIMARY_TRAINER.read_text()
    primary_model_source = PRIMARY_MODEL.read_text()
    secondary_runner_source = SECONDARY_RUNNER.read_text()
    primary_paths = sorted(O.glob("video_salmonn2_cmtb291_v1_shard[0-3].json"))
    primary_config_paths = sorted(
        (ROOT / "configs/models").glob("video_salmonn2_cmtb291_v1_shard[0-3].json")
    )
    primary_configs = [json.loads(path.read_text()) for path in primary_config_paths]
    primary = [row for path in primary_paths for row in json.loads(path.read_text())]
    primary_by_condition = Counter(row["condition"] for row in primary)
    primary_checks = {
        "four_frozen_input_shards": len(primary_paths) == 4,
        "raw_av_has_audio": all(
            "audio" in row and "timestamps" not in row
            for row in primary if row["condition"] == "Raw_AV"
        ),
        "full_video_only_has_no_audio_or_timestamps": all(
            "audio" not in row and "timestamps" not in row
            for row in primary if row["condition"] == "Video_only"
        ),
        "srvr_is_timestamped_video_only": all(
            "audio" not in row and len(row.get("timestamps", [])) == 2
            for row in primary if row["condition"] == "VAD_localized_clip"
        ),
        "matched_random_is_timestamped_video_only": all(
            "audio" not in row and len(row.get("timestamps", [])) == 2
            for row in primary if row["condition"].startswith("Matched_random")
        ),
        "primary_expected_counts": (
            primary_by_condition["Raw_AV"] == 291
            and primary_by_condition["Video_only"] == 291
            and primary_by_condition["VAD_localized_clip"] == 291
            and sum(
                count for condition, count in primary_by_condition.items()
                if condition.startswith("Matched_random")
            ) == 679
        ),
        "primary_preparer_adds_audio_only_for_raw": (
            'if condition == "Raw_AV":' in preparer_source
            and 'item["audio"] = row["video_path"]' in preparer_source
        ),
        "primary_loader_audio_decode_is_explicit_key_gated": (
            "if 'audio' in sources[0]:" in primary_loader_source
            and "audio_file = sources[0]['audio']" in primary_loader_source
            and "else:\n                audio_file = None" in primary_loader_source
        ),
        "primary_loader_no_audio_yields_video_modality_and_no_spectrogram": (
            "if 'audio' not in sources[0]:\n                    data_dict['modality'] = \"video\""
            in primary_loader_source
            and 'data_dict["spectrogram"] = None' in primary_loader_source
        ),
        "primary_collator_and_trainer_preserve_absent_spectrogram": (
            'if instances[0]["spectrogram"] is not None:' in primary_loader_source
            and 'if inputs.get("spectrogram") is not None:' in primary_trainer_source
            and 'inputs["spectrogram"] = None' in primary_trainer_source
        ),
        "primary_model_speech_encoder_is_audio_modality_and_spectrogram_gated": (
            'if spectrogram is not None and "audio" in modalities[0]:'
            in primary_model_source
            and "self.encode_speech(spectrogram)" in primary_model_source
        ),
        "primary_configs_are_audio_capable_but_sample_modality_is_key_gated": (
            len(primary_configs) == 4
            and all(config.get("audio_visual") is True for config in primary_configs)
        ),
    }

    secondary_path = O / "videollama21_av_eligibility24_v1.jsonl"
    secondary = jsonl(secondary_path)
    secondary_by_condition = Counter(row["condition"] for row in secondary)
    secondary_checks = {
        "raw_av_loader_has_audio": all(
            row["loader"]["va"] is True
            and row["loader"]["has_audio_tensor"] is True
            and row["audio_handling"]["source"]
            == "decoded_in_memory_from_original_video"
            for row in secondary if row["condition"] == "Raw_AV"
        ),
        "full_video_only_loader_has_no_audio": all(
            row["loader"]["va"] is False
            and row["loader"]["has_audio_tensor"] is False
            and row["loader"]["clip_start_sec"] is None
            and row["loader"]["clip_end_sec"] is None
            for row in secondary if row["condition"] == "Video_only"
        ),
        "srvr_loader_is_localized_video_only": all(
            row["loader"]["va"] is False
            and row["loader"]["has_audio_tensor"] is False
            and row["audio_handling"]["source"] == "disabled"
            and row["loader"]["clip_start_sec"] is not None
            and row["loader"]["clip_end_sec"] is not None
            for row in secondary if row["condition"] == "VAD_localized_clip"
        ),
        "secondary_expected_counts": (
            secondary_by_condition["Raw_AV"] == 44
            and secondary_by_condition["Video_only"] == 44
            and secondary_by_condition["VAD_localized_clip"] == 44
        ),
        "secondary_runner_sets_audio_only_for_raw_and_clip_only_for_srvr": (
            'va = condition == "Raw_AV"' in secondary_runner_source
            and 'clip = condition == "VAD_localized_clip"' in secondary_runner_source
            and 'kwargs = {"va": va}' in secondary_runner_source
        ),
    }

    checks = {**primary_checks, **secondary_checks}
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "intervention_contract": (
            "Speech timing selects a localized visual interval; the final SRVR model "
            "query is video-only on both evaluated backbones."
        ),
        "causal_boundary": (
            "Full-video Video-only controls audio removal alone. Same-duration "
            "non-overlapping video-only clips control removal plus generic shortening. "
            "The supported contrast is speech-timed visual evidence selection."
        ),
        "implementation_boundary": (
            "The primary trainer is configured with an audio-capable checkpoint, but its "
            "dataset loader decodes audio only when an explicit per-example audio key is "
            "present. Frozen SRVR and matched-random rows lack that key and therefore "
            "produce video modality with no spectrogram."
        ),
        "checks": checks,
        "primary_condition_counts": dict(sorted(primary_by_condition.items())),
        "secondary_condition_counts": dict(sorted(secondary_by_condition.items())),
        "input_sha256": {
            **{str(path.relative_to(ROOT)): sha256(path) for path in primary_paths},
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in primary_config_paths
            },
            str(secondary_path.relative_to(ROOT)): sha256(secondary_path),
            str(PRIMARY_PREPARER.relative_to(ROOT)): sha256(PRIMARY_PREPARER),
            str(PRIMARY_RUNNER.relative_to(ROOT)): sha256(PRIMARY_RUNNER),
            str(PRIMARY_LOADER.relative_to(ROOT)): sha256(PRIMARY_LOADER),
            str(PRIMARY_TRAINER.relative_to(ROOT)): sha256(PRIMARY_TRAINER),
            str(PRIMARY_MODEL.relative_to(ROOT)): sha256(PRIMARY_MODEL),
            str(SECONDARY_RUNNER.relative_to(ROOT)): sha256(SECONDARY_RUNNER),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    REPORT.write_text(
        "# SRVR input-modality audit v1\n\n"
        f"Status: **{result['status']}**\n\n"
        "The frozen implementation uses speech timing to select video frames, then "
        "queries each AVLLM without an audio tensor in the SRVR condition. Raw AV "
        "retains audio; full-video Video-only and every localized/random control do not.\n\n"
        "For video-SALMONN-2, the global configuration enables an audio-capable model, "
        "but the frozen dataset loader decodes container audio only when the example has "
        "an explicit `audio` key. Only Raw AV has that key; SRVR and matched-random rows "
        "produce `modality=video` and `spectrogram=None`. The audit hashes the frozen "
        "preparer, runner, loader, trainer, model modality gate, configs, and input "
        "shards. The collator/trainer preserve the absent spectrogram, and the model "
        "calls its speech encoder only when both a spectrogram and an audio modality "
        "are present.\n\n"
        "This makes the supported intervention **speech-timed visual evidence "
        "selection**, not joint audiovisual localization. Full-video Video-only "
        "shows that audio removal alone is insufficient, while equal-duration "
        "non-overlapping video-only clips show that removal plus arbitrary shortening "
        "is insufficient.\n\n"
        "## Checks\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in checks.items())
        + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
