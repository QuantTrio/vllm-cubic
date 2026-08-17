import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from tools import cubic_correctness_harness as HARNESS
from tools.pytest_cubic_shard import pytest_collection_modifyitems


def _position(token, logprob, top):
    return {"token": token, "logprob": logprob, "top_logprobs": top}


def test_compare_exact_accepts_identical_float_values():
    result = [_position((1,), -0.25, {(1,): -0.25, (2,): -1.0})]

    assert HARNESS.compare_exact(result, result) == []


def test_compare_exact_rejects_any_float_difference():
    reference = [_position((1,), -0.25, {(1,): -0.25, (2,): -1.0})]
    candidate = [_position((1,), -0.2500000001, {(1,): -0.25, (2,): -1.0})]

    errors = HARNESS.compare_exact(reference, candidate)

    assert errors == [
        "position 0: target logprob -0.25 != -0.2500000001"
    ]


def test_compare_exact_rejects_topk_set_and_token_difference():
    reference = [_position((1,), -0.25, {(1,): -0.25, (2,): -1.0})]
    candidate = [_position((9,), -0.25, {(1,): -0.25, (3,): -1.0})]

    errors = HARNESS.compare_exact(reference, candidate)

    assert errors == [
        "position 0: token (1,) != (9,)",
        "position 0: top-k token set differs",
    ]


def test_variant_matrix_rejects_batch_only_difference():
    baseline = [_position("token:a", -0.25, {"token:a": -0.25})]
    changed = [_position("token:a", -0.5, {"token:a": -0.5})]
    cases = {
        "short_fresh": baseline,
        "short_hit": baseline,
        "batch2_solo_0": baseline,
        "batch2_concurrent_0": baseline,
    }
    captures = {
        variant.name: dict(cases) for variant in HARNESS.VARIANTS
    }
    captures["mtp1_prefix_on"]["batch2_concurrent_0"] = changed

    errors = HARNESS.compare_variant_matrix(captures)

    assert any("batch invariance" in error for error in errors)


def test_variant_matrix_accepts_supported_subset():
    baseline = [_position("token:a", -0.25, {"token:a": -0.25})]
    cases = {
        "short_fresh": baseline,
        "short_hit": baseline,
        "batch2_solo_0": baseline,
        "batch2_concurrent_0": baseline,
    }
    captures = {
        "mtp0_prefix_off": dict(cases),
        "mtp0_prefix_on": dict(cases),
    }

    assert HARNESS.compare_variant_matrix(captures) == []


def test_source_fingerprint_changes_with_untracked_runtime_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    HARNESS.subprocess.run(("git", "init", "-q", str(repo)), check=True)
    HARNESS.subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "test@example.com"),
        check=True,
    )
    HARNESS.subprocess.run(
        ("git", "-C", str(repo), "config", "user.name", "Test"), check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("tracked")
    HARNESS.subprocess.run(
        ("git", "-C", str(repo), "add", "tracked.txt"), check=True
    )
    HARNESS.subprocess.run(
        ("git", "-C", str(repo), "commit", "-qm", "initial"), check=True
    )
    before = HARNESS.source_fingerprint(repo)
    runtime = repo / "vllm" / "runtime.py"
    runtime.parent.mkdir()
    runtime.write_text("VALUE = 1\n")

    assert HARNESS.source_fingerprint(repo) != before


def test_parse_cubic_case_requires_one_supported_cartesian_case():
    assert HARNESS.parse_cubic_case("5,A8,Linear,120") == (
        5,
        "A8",
        "Linear",
        120,
    )
    with pytest.raises(HARNESS.argparse.ArgumentTypeError):
        HARNESS.parse_cubic_case("5,A8,Linear,75")


def test_aggregate_coverage_only_uses_declared_passing_scope(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "source_fingerprint": "source",
                "failures": [],
                "correctness_contract": {"rtol": 0.0, "atol": 0.0},
                "audited_scope": {
                    "cubic_cases": [[3, "A16", "MoE", 90]],
                    "bf16_sms": [90],
                },
            }
        )
    )

    cubic_cases, bf16_sms = HARNESS.aggregate_coverage(
        "source",
        {(5, "A8", "Linear", 120)},
        {120},
        [report_path],
    )

    assert cubic_cases == {
        (3, "A16", "MoE", 90),
        (5, "A8", "Linear", 120),
    }
    assert bf16_sms == {90, 120}


def test_reused_bf16_captures_require_exactly_compatible_report(tmp_path):
    baseline = [_position("token:a", -0.25, {"token:a": -0.25})]
    variants = HARNESS.VARIANTS[:2]
    report_path = tmp_path / "report.json"
    report = {
        "source_fingerprint": "source",
        "failures": [],
        "correctness_contract": {"rtol": 0.0, "atol": 0.0},
        "bf16_model": "bf16",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8_q16",
        "seed": 7,
        "top_logprobs": 20,
        "capture_profile": "full",
        "capture_max_tokens": 128,
        "long_context_repetitions": 3600,
        "capture_images": [],
        "extra_serve_args": ["--limit-mm-per-prompt", '{"image":0}'],
        "captures": {
            "bf16": {variant.name: {"short_fresh": baseline} for variant in variants}
        },
    }
    report_path.write_text(json.dumps(report))
    args = Namespace(
        bf16_model="bf16",
        tensor_parallel_size=8,
        kv_cache_dtype="fp8_q16",
        seed=7,
        top_logprobs=20,
        capture_profile="full",
        capture_max_tokens=128,
        long_context_repetitions=3600,
        capture_image=[],
        extra_serve_arg=["--limit-mm-per-prompt", '{"image":0}'],
        extra_bf16_serve_arg=[],
    )

    captures = HARNESS.load_reused_bf16_captures(
        report_path, "source", args, variants
    )

    assert set(captures) == {variant.name for variant in variants}
    args.seed = 8
    with pytest.raises(RuntimeError, match="incompatible seed"):
        HARNESS.load_reused_bf16_captures(
            report_path, "source", args, variants
        )


def test_performance_is_blocked_by_any_missing_strict_case():
    with pytest.raises(RuntimeError, match="performance evaluation is blocked"):
        HARNESS.require_complete_coverage_for_performance(
            ["benchmark"],
            {(5, "A8", "Linear", 120)},
            {120},
        )

    missing = HARNESS.require_complete_coverage_for_performance(
        ["benchmark"],
        set(HARNESS.REQUIRED_CUBIC_CASES),
        set(HARNESS.SUPPORTED_CUDA_SMS),
    )
    assert missing == (set(), set())


def test_serve_command_always_enables_strict_eager_execution():
    args = Namespace(
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        seed=7,
        port=8000,
        extra_serve_arg=[],
        extra_cubic_serve_arg=[],
        extra_bf16_serve_arg=[],
    )

    command = HARNESS._serve_command(
        args,
        HARNESS.VARIANTS[0],
        case_name="cubic",
        model="model",
        served_model_name="served",
    )

    assert "--deterministic-inference" in command
    assert "--enforce-eager" in command


def test_serve_command_uses_only_the_selected_model_arguments():
    args = Namespace(
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        seed=7,
        port=8000,
        extra_serve_arg=["--common"],
        extra_cubic_serve_arg=["--cubic-only"],
        extra_bf16_serve_arg=["--reference-only"],
    )

    cubic = HARNESS._serve_command(
        args,
        HARNESS.VARIANTS[0],
        case_name="cubic",
        model="model",
        served_model_name="served",
    )
    reference = HARNESS._serve_command(
        args,
        HARNESS.VARIANTS[0],
        case_name="bf16",
        model="model",
        served_model_name="served",
    )

    assert "--common" in cubic and "--common" in reference
    assert "--cubic-only" in cubic and "--cubic-only" not in reference
    assert "--reference-only" in reference and "--reference-only" not in cubic


def test_skip_bf16_is_fail_closed_for_bf16_coverage(monkeypatch, tmp_path):
    common = [
        "harness",
        "--model",
        "cubic",
        "--served-model-name",
        "cubic",
        "--tensor-parallel-size",
        "1",
        "--cuda-visible-devices",
        "0",
        "--artifact-dir",
        str(tmp_path),
        "--skip-bf16",
    ]
    monkeypatch.setattr(sys, "argv", common)
    assert HARNESS.parse_args().bf16_model is None

    monkeypatch.setattr(sys, "argv", common + ["--audit-bf16-sm", "120"])
    with pytest.raises(SystemExit):
        HARNESS.parse_args()


def test_capture_checkpoint_reuses_only_an_exact_identity(tmp_path):
    path = tmp_path / "checkpoint.json"
    captures = {"mtp0_prefix_off": {"short_fresh": []}}
    identity = {"source_fingerprint": "source", "seed": 7}
    cases = {"cubic": {"identity": identity, "captures": captures}}

    HARNESS._write_capture_checkpoint(path, cases)

    assert HARNESS._load_capture_checkpoint(path) == cases


def test_capture_checkpoint_keeps_model_cases_independent(tmp_path):
    path = tmp_path / "checkpoint.json"
    cases = {
        "cubic": {
            "identity": {"model": "cubic", "extra_case_serve_args": []},
            "captures": {"mtp0_prefix_off": {"short_fresh": []}},
        },
        "bf16": {
            "identity": {"model": "native", "extra_case_serve_args": []},
            "captures": {},
        },
    }

    HARNESS._write_capture_checkpoint(path, cases)
    loaded = HARNESS._load_capture_checkpoint(path)
    loaded["bf16"]["identity"]["extra_case_serve_args"] = ["--moe-backend"]

    assert loaded["cubic"] == cases["cubic"]
    assert loaded["bf16"] != cases["bf16"]


def test_capture_identity_does_not_invalidate_completed_variants():
    args = Namespace(
        tensor_parallel_size=8,
        cuda_visible_devices="0,1,2,3,4,5,6,7",
        kv_cache_dtype="fp8_q16",
        seed=7,
        top_logprobs=20,
        capture_profile="full",
        capture_max_tokens=128,
        long_context_repetitions=3600,
        capture_image=[],
        variants=HARNESS.VARIANTS[:1],
        extra_serve_arg=[],
        extra_cubic_serve_arg=[],
        extra_bf16_serve_arg=[],
    )
    before = HARNESS._capture_checkpoint_identity(
        args,
        "source",
        case_name="cubic",
        model="model",
        served_model_name="served",
    )
    args.variants = HARNESS.VARIANTS
    after = HARNESS._capture_checkpoint_identity(
        args,
        "source",
        case_name="cubic",
        model="model",
        served_model_name="served",
    )

    assert before == after


def test_capture_one_reuses_rendered_tokens_with_unique_request_id(monkeypatch):
    calls = []
    response = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "a",
                            "bytes": [97],
                            "logprob": -0.25,
                            "top_logprobs": [
                                {
                                    "token": "a",
                                    "bytes": [97],
                                    "logprob": -0.25,
                                }
                            ],
                        }
                    ]
                }
            }
        ]
    }

    def request(url, payload=None):
        calls.append((url, payload))
        return response

    monkeypatch.setattr(HARNESS, "_request_json", request)
    rendered = {"token_ids": [1, 2, 3], "sampling_params": {"seed": 7}}

    first = HARNESS._capture_one("http://server", rendered)
    second = HARNESS._capture_one("http://server", rendered)

    assert first == second
    assert calls[0][0].endswith("/inference/v1/generate")
    assert calls[0][1]["token_ids"] == [1, 2, 3]
    assert calls[0][1]["request_id"] != calls[1][1]["request_id"]
    assert "request_id" not in rendered


def test_render_prompt_preserves_explicit_generation_bound(monkeypatch):
    monkeypatch.setattr(
        HARNESS,
        "_request_json",
        lambda url, payload=None: {
            "token_ids": [1, 2, 3],
            "sampling_params": {"temperature": 0.0},
        },
    )
    args = Namespace(
        capture_max_tokens=16,
        seed=7,
        top_logprobs=20,
    )

    rendered = HARNESS._render_prompt(
        "http://server", args, "prompt", "served"
    )

    assert rendered["sampling_params"]["max_tokens"] == 16


def test_breadth_capture_stops_before_long_context_and_c8(monkeypatch):
    monkeypatch.setattr(
        HARNESS,
        "_render_prompt",
        lambda base_url, args, prompt, served_model_name: {"prompt": prompt},
    )
    monkeypatch.setattr(
        HARNESS,
        "_capture_one",
        lambda base_url, rendered_request: [
            {"token": rendered_request["prompt"]}
        ],
    )
    args = Namespace(capture_profile="breadth", long_context_repetitions=3600)

    captures = HARNESS.capture_variant("http://server", args, "served")

    assert set(captures) == {
        "short_fresh",
        "short_hit",
        "batch2_solo_0",
        "batch2_solo_1",
        "batch2_concurrent_0",
        "batch2_concurrent_1",
    }


def test_multimodal_capture_covers_repeated_and_distinct_images(
    monkeypatch, tmp_path
):
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    images[0].write_bytes(b"first")
    images[1].write_bytes(b"second")
    monkeypatch.setattr(
        HARNESS,
        "_render_prompt",
        lambda base_url, args, prompt, served_model_name: {"prompt": prompt},
    )
    monkeypatch.setattr(
        HARNESS,
        "_capture_one",
        lambda base_url, rendered_request: [
            {"token": str(rendered_request["prompt"])}
        ],
    )
    args = Namespace(
        capture_profile="multimodal",
        capture_image=images,
        long_context_repetitions=3600,
    )

    captures = HARNESS.capture_variant("http://server", args, "served")

    assert set(captures) == {
        "image0_fresh",
        "image0_hit",
        "image1_fresh",
        "image1_hit",
        "mm_same_batch2_solo_0",
        "mm_same_batch2_solo_1",
        "mm_same_batch2_concurrent_0",
        "mm_same_batch2_concurrent_1",
        "mm_distinct_batch2_solo_0",
        "mm_distinct_batch2_solo_1",
        "mm_distinct_batch2_concurrent_0",
        "mm_distinct_batch2_concurrent_1",
    }


def test_capture_image_metadata_is_content_addressed(tmp_path):
    image = tmp_path / "fixture.png"
    image.write_bytes(b"first")
    before = HARNESS._capture_image_metadata([image])
    image.write_bytes(b"second")
    after = HARNESS._capture_image_metadata([image])

    assert before[0]["suffix"] == ".png"
    assert before[0]["sha256"] != after[0]["sha256"]


def test_correctness_shards_partition_collection_without_overlap():
    deselected = []
    items = list(range(10))
    config = SimpleNamespace(
        getoption=lambda option: {
            "--cubic-shard-count": 3,
            "--cubic-shard-index": 1,
        }[option],
        hook=SimpleNamespace(
            pytest_deselected=lambda *, items: deselected.extend(items)
        ),
    )

    pytest_collection_modifyitems(config, items)

    assert items == [1, 4, 7]
    assert sorted(items + deselected) == list(range(10))
