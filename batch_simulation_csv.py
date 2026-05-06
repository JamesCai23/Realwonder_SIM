import argparse
import os
from typing import Dict, List, Sequence

import numpy as np

from batch_pipeline_csv import (
    _build_job,
    _build_sim_config,
    _is_complete_final_sim,
    _normalize_stage_list,
    _t2i_generate_images,
    _validate_stages,
    _write_yaml,
    load_jobs,
)
from case_simulation import preload_case_simulation_models, run_case_simulation
from generate_quantiphy_qa import generate_csv


def _validate_sim_stages(stages: Sequence[str]) -> None:
    allowed = {"t2i", "i2s", "sim", "qa"}
    bad = [s for s in stages if s not in allowed]
    if bad:
        raise ValueError(f"Unknown stage(s): {bad}. Allowed: {sorted(allowed)}")


def _is_complete_sim_output(sim_data_path: str) -> bool:
    if not _is_complete_final_sim(sim_data_path):
        return False

    sim_root = os.path.dirname(sim_data_path)
    kinematics_path = os.path.join(sim_root, "simulation", "kinematics_log.json")
    simulation_mp4 = os.path.join(sim_data_path, "simulation.mp4")
    if not os.path.exists(kinematics_path) or os.path.getsize(kinematics_path) <= 0:
        return False
    if not os.path.exists(simulation_mp4) or os.path.getsize(simulation_mp4) <= 0:
        return False
    return True


def _load_rgb_image(path: str):
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return None

    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_grayscale_mask(path: str):
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return None

    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _resolve_inpainter_mask_path(job: Dict[str, object]) -> str:
    sim_data_path = str(job["sim_data_path"])
    case_output_dir = str(job["case_output_dir"])
    candidate_paths = [
        os.path.join(os.path.dirname(sim_data_path), "inpainter_masks.png"),
        os.path.join(case_output_dir, "inpainter_masks.png"),
    ]
    for path in candidate_paths:
        print(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return candidate_paths[0]


def _is_all_black_mask(mask: np.ndarray) -> bool:
    return mask.size == 0 or not np.any(mask > 0)


def _mean_abs_diff(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if image_a.shape != image_b.shape:
        return float("inf")

    diff = np.abs(image_a.astype(np.float32) - image_b.astype(np.float32))
    if mask is not None:
        if mask.shape[:2] != diff.shape[:2]:
            return float("inf")
        mask_bool = mask > 0
        if not mask_bool.any():
            return float("inf")
        diff = diff[mask_bool]
    return float(diff.mean()) if diff.size else float("inf")


def _video_motion_score(video_path: str) -> float | None:
    if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
        return None

    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio

    reader = imageio.get_reader(video_path)
    try:
        previous_frame = None
        diffs = []
        frame_count = 0
        for frame in reader:
            frame_array = np.asarray(frame, dtype=np.float32)
            if frame_array.ndim == 2:
                frame_array = np.repeat(frame_array[..., None], 3, axis=2)
            elif frame_array.shape[-1] > 3:
                frame_array = frame_array[..., :3]

            if previous_frame is not None:
                diffs.append(float(np.mean(np.abs(frame_array - previous_frame))))
            previous_frame = frame_array
            frame_count += 1

        if frame_count < 2 or not diffs:
            return None
        return float(np.mean(diffs))
    finally:
        reader.close()


def _flow_quality_score(flow_path: str) -> float | None:
    if not os.path.exists(flow_path) or os.path.getsize(flow_path) <= 0:
        return None

    try:
        flow = np.load(flow_path)
    except Exception:
        return None

    if flow.ndim != 4 or flow.shape[1] != 2:
        return None

    flow = np.asarray(flow, dtype=np.float32)
    if not np.isfinite(flow).all():
        return None

    magnitude = np.sqrt(np.sum(flow * flow, axis=1))
    return float(np.mean(magnitude))


def _collect_strict_quality_reasons(job: Dict[str, object]) -> List[str]:
    reasons: List[str] = []
    case_output_dir = str(job["case_output_dir"])
    sim_data_path = str(job["sim_data_path"])

    if not _is_complete_sim_output(sim_data_path):
        reasons.append("incomplete final_sim output")
        return reasons

    mask_path = _resolve_inpainter_mask_path(job)
    mask = _load_grayscale_mask(mask_path)
    if mask is None:
        reasons.append("inpaint failed: missing inpainter_masks.png")
    elif _is_all_black_mask(mask):
        reasons.append("inpaint failed: inpainter_masks.png is all black")

    video_motion_score = _video_motion_score(os.path.join(sim_data_path, "simulation.mp4"))
    if video_motion_score is None:
        reasons.append("unreadable simulation.mp4")
    elif video_motion_score < 0.5:
        reasons.append(f"simulation.mp4 is too static (motion score={video_motion_score:.4f})")

    flow_score = _flow_quality_score(os.path.join(sim_data_path, "flows.npy"))
    if flow_score is None:
        reasons.append("missing or invalid flows.npy")
    elif flow_score < 0.05:
        reasons.append(f"optical flow magnitude is too small (score={flow_score:.4f})")

    return reasons


def _collect_mask_only_reasons(job: Dict[str, object]) -> List[str]:
    reasons: List[str] = []

    mask_path = _resolve_inpainter_mask_path(job)
    mask = _load_grayscale_mask(mask_path)
    if mask is None:
        reasons.append("inpaint failed: missing inpainter_masks.png")
    elif _is_all_black_mask(mask):
        reasons.append("inpaint failed: inpainter_masks.png is all black")

    return reasons


def _collect_qa_filter_reasons(job: Dict[str, object], strategy: str) -> List[str]:
    normalized_strategy = (strategy or "mask_only").strip().lower()
    if normalized_strategy == "none":
        return []
    if normalized_strategy == "mask_only":
        return _collect_mask_only_reasons(job)
    raise ValueError(f"Unknown qa_filter_strategy: {strategy}. Allowed: ['mask_only', 'strict', 'none']")


def _collect_bad_case_reasons(jobs: List[Dict[str, object]], strategy: str) -> Dict[str, List[str]]:
    bad_cases: Dict[str, List[str]] = {}
    for job in jobs:
        reasons = _collect_qa_filter_reasons(job, strategy=strategy)
        if reasons:
            bad_cases[str(job["case_name"])] = reasons
    return bad_cases


def _ensure_i2s_config(job: Dict[str, object], args, dry_run: bool) -> None:
    config_path = str(job["config_path"])
    if (not args.overwrite) and os.path.exists(config_path) and os.path.getsize(config_path) > 0:
        print(f"[{job['idx']}/{args.total}] i2s case={job['case_name']} (config.yaml exists, skip)")
        return

    input_path = os.path.join(str(job["case_output_dir"]), "input.png")
    if (not dry_run) and (not os.path.exists(input_path)):
        raise FileNotFoundError(
            f"Missing input image for i2s: {input_path}. "
            "Run with --stages t2i,i2s first, or provide the image."
        )

    sim_config = _build_sim_config(
        case_name=str(job["case_name"]),
        case_output_dir=str(job["case_output_dir"]),
        result_root=args.result_root,
        seed=int(job["seed"]),
        object_name=str(job["object_name"]),
        img_prompt=str(job["image_prompt"]),
        vgen_prompt=str(job["vgen_prompt"]),
        motion_mode=str(job["motion_mode"]),
        motion_direction=job["motion_direction"],
        force_direction=job["force_direction"],
        force_strength=job["force_strength"],
        gravity=job["gravity"],
        initial_velocity=job["initial_velocity"],
        simulated_frames_num=job["simulated_frames_num"],
        estimated_real_size=job["estimated_real_size"],
    )

    if not dry_run:
        _write_yaml(config_path, sim_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch simulation pipeline from CSV (sim-only):\n"
            "  t2i: text-to-image (generate input.png via vLLM)\n"
            "  i2s: image-to-scene (write config.yaml)\n"
            "  sim: run physics simulation and export final_sim artifacts\n"
            "  qa: generate QA CSV from simulation kinematics\n\n"
            "No WAN model video generation is involved."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--output_root", type=str, default="cases", help="Default case output root")
    parser.add_argument("--result_root", type=str, default="result", help="Default result root")
    parser.add_argument("--sim_output_dir", type=str, default="000", help="Simulation output sub-folder")
    parser.add_argument("--seed", type=int, default=42, help="Default seed")
    parser.add_argument("--eval_degradation", type=float, default=0.05, help="Compatibility field for CSV parser")
    parser.add_argument("--franka_step", type=int, default=0, help="Compatibility field for CSV parser")
    parser.add_argument("--mask_dropin_step", type=int, default=1, help="Compatibility field for CSV parser")
    parser.add_argument(
        "--denoising_step_list",
        type=str,
        default="[900, 600, 300, 100]",
        help="Compatibility field for CSV parser",
    )
    parser.add_argument(
        "--scene_model_name",
        type=str,
        default="/home/lff/bigdata1/huggingface/Qwen-Image",
        help="Qwen-Image model path for t2i",
    )
    parser.add_argument(
        "--t2i_base_url",
        type=str,
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM OpenAI-compatible base_url for image generation",
    )
    parser.add_argument(
        "--t2i_fallback_root",
        type=str,
        default="/home/lff/data1/cym/physical_data/RealWonder/cases",
        help="Fallback root containing <case_name>/input.png when t2i endpoint is unavailable",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated stages: t2i,i2s,sim,qa (or 'all'). Default runs all.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print planned work only")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--qa_output_csv",
        type=str,
        default="",
        help="Output QA CSV path (default: <result_root>/quantiphy_synthetic_dataset.csv)",
    )
    parser.add_argument(
        "--qa_filter_strategy",
        type=str,
        default="mask_only",
        choices=["mask_only", "none"],
        help="QA filtering strategy: mask_only (default), strict, or none",
    )

    args, _ = parser.parse_known_args()

    stages = _normalize_stage_list(args.stages)
    if not stages:
        stages = ["t2i", "i2s", "sim", "qa"]
    _validate_stages([s for s in stages if s in {"t2i", "i2s"}])
    _validate_sim_stages(stages)

    rows = load_jobs(args.csv_path)
    jobs: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        job = _build_job(row=row, idx=idx, args=args)
        if "t2i" in stages and not job["image_prompt"]:
            raise ValueError(f"Row {idx}: image_prompt is required when stage t2i is enabled")
        jobs.append(job)

    total = len(jobs)
    args.total = total

    for stage_idx, stage in enumerate(stages, start=1):
        if stage == "t2i":
            print(f"Stage {stage_idx}/{len(stages)} t2i (text-to-image): {total} cases")
            if args.dry_run:
                for job in jobs:
                    out_png = os.path.join(str(job["case_output_dir"]), "input.png")
                    print(f"[{job['idx']}/{total}] t2i case={job['case_name']} -> {out_png}")
                continue

            _t2i_generate_images(
                model_name=args.scene_model_name,
                jobs=jobs,
                total=total,
                overwrite=args.overwrite,
                base_url=args.t2i_base_url,
                fallback_root=args.t2i_fallback_root,
            )

        elif stage == "i2s":
            print(f"Stage {stage_idx}/{len(stages)} i2s (image-to-scene): {total} cases")
            for job in jobs:
                print(f"[{job['idx']}/{total}] i2s case={job['case_name']}")
                _ensure_i2s_config(job, args=args, dry_run=args.dry_run)

        elif stage == "sim":
            print(f"Stage {stage_idx}/{len(stages)} sim (physics simulation): {total} cases")
            preload_case_simulation_models(
                {
                    "device": os.environ.get("REALWONDER_DEVICE", "cuda:0"),
                    "sam3_device": os.environ.get("REALWONDER_SAM3_DEVICE", os.environ.get("REALWONDER_DEVICE", "cuda:0")),
                    "inpaint_device": os.environ.get("REALWONDER_INPAINT_DEVICE", os.environ.get("REALWONDER_DEVICE", "cuda:0")),
                    "noise_device": os.environ.get("REALWONDER_NOISE_DEVICE", os.environ.get("REALWONDER_DEVICE", "cuda:0")),
                    "preload_inpaint": os.environ.get("REALWONDER_PRELOAD_INPAINT", "1") not in {"0", "false", "False"},
                    "release_inpaint_before_noise": os.environ.get("REALWONDER_RELEASE_INPAINT_BEFORE_NOISE", "0") in {"1", "true", "True"},
                }
            )
            for job in jobs:
                sim_data_path = str(job["sim_data_path"])
                if (not args.overwrite) and _is_complete_sim_output(sim_data_path):
                    print(f"[{job['idx']}/{total}] sim case={job['case_name']} (final_sim complete, skip)")
                    continue

                print(f"[{job['idx']}/{total}] sim case={job['case_name']}")
                if args.dry_run:
                    continue

                _ensure_i2s_config(job, args=args, dry_run=False)
                run_case_simulation(
                    config_path=str(job["config_path"]),
                    output_dir=str(job["sim_output_dir"]),
                )

        elif stage == "qa":
            qa_output_csv = args.qa_output_csv.strip() if args.qa_output_csv else ""
            if not qa_output_csv:
                qa_output_csv = os.path.join(args.result_root, "quantiphy_synthetic_dataset.csv")

            print(
                f"Stage {stage_idx}/{len(stages)} qa (generate QA CSV, filter={args.qa_filter_strategy}): {qa_output_csv}"
            )
            if args.dry_run:
                continue

            bad_case_reasons = _collect_bad_case_reasons(jobs, strategy=args.qa_filter_strategy)
            if bad_case_reasons:
                print(f"Quality filter excluded {len(bad_case_reasons)} case(s):")
                for case_name, reasons in bad_case_reasons.items():
                    print(f"  - {case_name}: {', '.join(reasons)}")

            os.makedirs(os.path.dirname(qa_output_csv) or ".", exist_ok=True)
            skip_case_names = None if args.qa_filter_strategy == "mask_only" and not bad_case_reasons else list(bad_case_reasons.keys())
            generate_csv(
                args.result_root,
                qa_output_csv,
                skip_case_names=skip_case_names,
            )

    print(f"All done. Processed {total} cases. stages={','.join(stages)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
