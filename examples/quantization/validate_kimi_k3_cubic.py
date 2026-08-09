# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from vllm import LLM, SamplingParams

PROMPT = "背诵千字文全文"
QIANZIWEN_ANCHORS = (
    "天地玄黄",
    "寒来暑往",
    "云腾致雨",
    "金生丽水",
    "龙师火帝",
    "坐朝问道",
    "罔谈彼短",
    "矩步引领",
    "谓语助者",
    "焉哉乎也",
)


def validate_mixed_precision_config(model: Path) -> tuple[list[dict], float]:
    with (model / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    text_config = config.get("text_config", config)
    quantization = text_config.get("quantization_config", {})
    schedule = quantization.get("layer_bit_schedule", [])
    effective_bits = float(
        quantization.get("converted_tensor_effective_bits", float("inf"))
    )
    tensor_overrides = quantization.get("tensor_bit_overrides", [])
    widths = [int(rule["num_bits"]) for rule in schedule]
    widths.extend(int(rule["num_bits"]) for rule in tensor_overrides)
    layer_widths: dict[int, int] = {}
    for rule in schedule:
        for layer in range(
            int(rule["start_layer"]),
            int(rule["end_layer"]) + 1,
        ):
            if layer in layer_widths:
                raise ValueError(f"Kimi-K3 layer {layer} has overlapping bit rules.")
            layer_widths[layer] = int(rule["num_bits"])
    if (
        quantization.get("runtime_weight_storage") != "native_packed_bitstream"
        or set(widths) != set(range(1, 9))
        or any(bits not in range(1, 9) for bits in widths)
        or set(layer_widths) != set(range(1, 93))
        or layer_widths[92] != 4
        or effective_bits > 2.5
    ):
        raise ValueError(
            "Kimi-K3 validation requires every width from 1 through 8, "
            "layers 1--92 exactly once, layer 92 at 4-bit, native packed "
            "runtime storage, and <=2.5 effective bits."
        )
    return schedule, effective_bits


def has_repetition_loop(text: str, ngram_size: int = 32) -> bool:
    compact = "".join(text.split())
    if len(compact) < ngram_size * 3:
        return False
    ngrams = Counter(
        compact[index : index + ngram_size]
        for index in range(len(compact) - ngram_size + 1)
    )
    return any(count >= 3 for count in ngrams.values())


def recited_qianziwen(text: str) -> bool:
    compact = "".join(
        character for character in text if "\u4e00" <= character <= "\u9fff"
    )
    return len(compact) >= 950 and all(
        anchor in compact for anchor in QIANZIWEN_ANCHORS
    )


def summarize_routes(routes: np.ndarray) -> dict[str, float | int | list[int]]:
    routed_layer_mask = np.any(routes != 0, axis=(0, 2))
    routed = routes[:, routed_layer_mask]
    if routed.shape[1] == 0:
        raise ValueError("Routed-expert capture contains no MoE layer data.")
    linear_ranks = routed // 112
    round_robin_ranks = routed % 8

    def placement_stats(ranks: np.ndarray) -> tuple[float, float, float]:
        counts = np.stack(
            [(ranks == rank).sum(axis=-1) for rank in range(8)],
            axis=-1,
        )
        max_routes = counts.max(axis=-1)
        return (
            float(max_routes.mean()),
            float(np.quantile(max_routes, 0.95)),
            float((2.0 / max_routes).mean()),
        )

    linear_mean, linear_p95, linear_balancedness = placement_stats(linear_ranks)
    rr_mean, rr_p95, rr_balancedness = placement_stats(round_robin_ranks)
    return {
        "tokens": int(routes.shape[0]),
        "layers": int(routes.shape[1]),
        "moe_layers": int(routed.shape[1]),
        "top_k": int(routes.shape[2]),
        "linear_mean_critical_routes": linear_mean,
        "linear_p95_critical_routes": linear_p95,
        "linear_balancedness": linear_balancedness,
        "round_robin_mean_critical_routes": rr_mean,
        "round_robin_p95_critical_routes": rr_p95,
        "round_robin_balancedness": rr_balancedness,
    }


def generate_streaming(
    llm: LLM,
    sampling_params: SamplingParams,
    live_output: Path,
):
    engine = llm.llm_engine
    messages = [{"role": "user", "content": PROMPT}]
    prompt_token_ids = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    print(
        json.dumps(
            {
                "event": "validation_request",
                "prompt": PROMPT,
                "request_format": "kimi_k3_chat_template_default_thinking",
                "prompt_tokens": len(prompt_token_ids),
                "sampling_params": {
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                },
                "live_output": str(live_output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    engine.add_request(
        "cubic-kimi-k3-validation",
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
    )
    live_output.parent.mkdir(parents=True, exist_ok=True)
    live_output.write_text("", encoding="utf-8")
    print("[model_output_begin]", flush=True)

    result = None
    previous_text = ""
    started_at = time.monotonic()
    last_status_at = started_at
    while engine.has_unfinished_requests():
        for request_output in engine.step():
            if not request_output.outputs:
                continue
            result = request_output.outputs[0]
            current_text = result.text
            if current_text.startswith(previous_text):
                print(current_text[len(previous_text) :], end="", flush=True)
            else:
                print(f"\n[output snapshot]\n{current_text}", end="", flush=True)
            previous_text = current_text
            live_output.write_text(current_text, encoding="utf-8")

            now = time.monotonic()
            if now - last_status_at >= 10 or request_output.finished:
                elapsed = now - started_at
                num_tokens = len(result.token_ids)
                print(
                    (
                        f"\n[progress] tokens={num_tokens} "
                        f"elapsed={elapsed:.1f}s "
                        f"speed={num_tokens / elapsed:.3f} tok/s"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_status_at = now
    print(flush=True)
    if result is None:
        raise RuntimeError("The engine finished without returning model output.")
    print("[model_output_end]", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.985)
    parser.add_argument(
        "--kv-cache-memory-gib",
        type=float,
        default=0,
        help="Explicit per-GPU KV-cache budget; set to 0 for automatic sizing.",
    )
    parser.add_argument(
        "--dspark-model",
        type=Path,
        help="Optional Kimi-K3 DSpark model for speculative decoding.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=7,
        help="Number of DSpark draft tokens proposed per target-model step.",
    )
    parser.add_argument(
        "--live-output",
        type=Path,
        default=Path("logs/kimi-k3-cubic-live-output.txt"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optionally capture ten steady-state decode iterations.",
    )
    parser.add_argument(
        "--routes-output",
        type=Path,
        help="Optionally capture generated-token expert routes as NumPy data.",
    )
    parser.add_argument(
        "--cudagraph-mode",
        choices=("NONE", "PIECEWISE", "FULL", "FULL_AND_PIECEWISE"),
        help="Override the CUDA graph mode for correctness diagnostics.",
    )
    args = parser.parse_args()
    schedule, effective_bits = validate_mixed_precision_config(args.model)
    with (args.model / "config.json").open(encoding="utf-8") as file:
        model_config = json.load(file)
    text_config = model_config.get("text_config", model_config)
    placement = text_config["quantization_config"].get("expert_placement", {})
    num_redundant_experts = int(placement.get("num_redundant_experts", 0))
    enable_eplb = num_redundant_experts > 0
    eplb_config = None
    if enable_eplb:
        eplb_config = {
            "num_redundant_experts": num_redundant_experts,
            "window_size": 1,
            "step_interval": 100000,
            "log_balancedness": False,
            "use_async": False,
        }
    print(
        json.dumps(
            {
                "event": "expert_replica_config",
                "enable_eplb": enable_eplb,
                "num_redundant_experts": num_redundant_experts,
                "kv_cache_memory_gib": args.kv_cache_memory_gib,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    speculative_config = None
    if args.dspark_model is not None:
        speculative_config = {
            "method": "dspark",
            "model": str(args.dspark_model),
            "num_speculative_tokens": args.num_speculative_tokens,
            "attention_backend": "FLASH_ATTN_MLA",
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
        }
    profiler_config = None
    if args.profile_dir is not None:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.absolute()),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_record_shapes": True,
            "ignore_frontend": True,
            "delay_iterations": 20,
            "max_iterations": 10,
        }
    eplb_kwargs = (
        {"enable_eplb": True, "eplb_config": eplb_config}
        if eplb_config is not None
        else {"enable_eplb": False}
    )
    llm = LLM(
        model=str(args.model),
        quantization="cubic",
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        **eplb_kwargs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=(
            int(args.kv_cache_memory_gib * (1 << 30))
            if args.kv_cache_memory_gib > 0
            else None
        ),
        seed=0,
        trust_remote_code=True,
        mm_encoder_tp_mode="data",
        enable_prefix_caching=True,
        attention_config={"use_prefill_query_quantization": True},
        profiler_config=profiler_config,
        enable_return_routed_experts=args.routes_output is not None,
        speculative_config=speculative_config,
        disable_log_stats=False,
        compilation_config=(
            {"cudagraph_mode": args.cudagraph_mode}
            if args.cudagraph_mode is not None
            else None
        ),
    )
    if profiler_config is not None:
        print(
            json.dumps(
                {
                    "event": "profiler_start",
                    "profile_dir": str(args.profile_dir),
                    "delay_iterations": 20,
                    "max_iterations": 10,
                }
            ),
            flush=True,
        )
        llm.start_profile("kimi-k3-cubic-decode")
    result = generate_streaming(
        llm,
        SamplingParams(
            max_tokens=args.max_tokens,
            stop=["<|close|>message<|sep|>"],
        ),
        args.live_output,
    )
    if profiler_config is not None:
        llm.stop_profile()
    if args.routes_output is not None:
        if result.routed_experts is None:
            raise RuntimeError("The engine did not return routed expert IDs.")
        args.routes_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.routes_output, result.routed_experts)
        print(
            json.dumps(
                {
                    "event": "routed_experts",
                    "output": str(args.routes_output),
                    **summarize_routes(result.routed_experts),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    loops = has_repetition_loop(result.text)
    stopped = result.finish_reason == "stop"
    recited = recited_qianziwen(result.text)
    verdict = {
        "prompt": PROMPT,
        "finish_reason": result.finish_reason,
        "recited_qianziwen": recited,
        "not_repeating": not loops,
        "stopped_normally": stopped,
        "passed": recited and not loops and stopped,
        "layer_bit_schedule": schedule,
        "effective_bits_per_weight": effective_bits,
        "text": result.text,
    }
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if not verdict["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
