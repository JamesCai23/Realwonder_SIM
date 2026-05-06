import argparse
import csv
import gc
import os
import shutil
from typing import Dict, List, Optional, Sequence


def _parse_vec6(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    values = [float(x) for x in text.replace(",", " ").split() if x]
    if len(values) != 6:
        raise ValueError(f"initial_velocity expects 6 values, got: {value}")
    return values


def _parse_vector3(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"Expected 3 values, got {value}")
        return [float(v) for v in value]

    text = str(value).strip()
    if not text:
        return list(default)

    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) == 3:
        return [float(p) for p in parts]

    raise ValueError(f"Cannot parse 3D vector from: {value}")


def load_jobs(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    return rows


def _build_job(row: Dict[str, str], idx: int, args) -> Dict[str, object]:
    case_name = (row.get("case_name") or "").strip()
    if not case_name:
        raise ValueError(f"Row {idx}: case_name is required")

    object_name = (row.get("object_name") or case_name).strip()
    image_prompt = (row.get("image_prompt") or row.get("img_prompt") or "").strip()
    motion_mode = (row.get("motion_mode") or "constant_velocity").strip()

    case_output_dir = (row.get("output_dir") or os.path.join(args.output_root, case_name)).strip()
    sim_output_dir = (row.get("sim_output_dir") or args.sim_output_dir).strip()
    seed = int(row.get("seed") or args.seed)

    motion_direction = _parse_vector3(row.get("motion_direction"), default=[1.0, 0.0, 0.0])
    force_direction = (
        _parse_vector3(row.get("force_direction"), default=[0.0, 0.0, 0.0]) if row.get("force_direction") else None
    )
    force_strength = float(row["force_strength"]) if row.get("force_strength") not in (None, "") else None
    gravity = float(row["gravity"]) if row.get("gravity") not in (None, "") else None
    initial_velocity = _parse_vec6(row.get("initial_velocity"))
    vgen_prompt = (row.get("vgen_prompt") or "").strip()
    estimated_real_size = row.get("estimated_real_size")

    config_path = row.get("config_path")
    if not config_path:
        config_path = os.path.join(case_output_dir, "config.yaml")

    sim_data_path = row.get("sim_data_path")
    if not sim_data_path:
        sim_data_path = os.path.join(args.result_root, case_name, sim_output_dir, "final_sim")

    output_path = row.get("infer_output_path")
    if not output_path:
        output_path = os.path.join(args.result_root, case_name, sim_output_dir, "final.mp4")

    eval_degradation = float(row.get("eval_degradation") or args.eval_degradation)
    franka_step = int(row.get("franka_step") or args.franka_step)
    mask_dropin_step = int(row.get("mask_dropin_step") or args.mask_dropin_step)
    denoising_step_list = row.get("denoising_step_list") or args.denoising_step_list
    simulated_frames_num = int(row.get("simulated_frames_num") or 51)

    return {
        "idx": idx,
        "case_name": case_name,
        "object_name": object_name,
        "image_prompt": image_prompt,
        "motion_mode": motion_mode,
        "case_output_dir": case_output_dir,
        "sim_output_dir": sim_output_dir,
        "seed": seed,
        "motion_direction": motion_direction,
        "force_direction": force_direction,
        "force_strength": force_strength,
        "gravity": gravity,
        "initial_velocity": initial_velocity,
        "vgen_prompt": vgen_prompt,
        "config_path": config_path,
        "sim_data_path": sim_data_path,
        "output_path": output_path,
        "eval_degradation": eval_degradation,
        "franka_step": franka_step,
        "mask_dropin_step": mask_dropin_step,
        "denoising_step_list": denoising_step_list,
        "simulated_frames_num": simulated_frames_num,
        "estimated_real_size": estimated_real_size,
    }


def _normalize_stage_list(stages: str) -> List[str]:
    text = (stages or "").strip()
    if not text:
        return []
    if text.lower() in {"all", "*"}:
        return ["t2i", "i2s", "s2v"]
    return [s.strip().lower() for s in text.split(",") if s.strip()]


def _validate_stages(stages: Sequence[str]) -> None:
    allowed = {"t2i", "i2s", "s2v"}
    bad = [s for s in stages if s not in allowed]
    if bad:
        raise ValueError(f"Unknown stage(s): {bad}. Allowed: {sorted(allowed)}")


def _is_complete_final_sim(sim_data_path: str) -> bool:
    if not os.path.isdir(sim_data_path):
        return False

    required_files = [
        "noises.npy",
        "resized_input_image.png",
        "prompt.txt",
        "config.yaml",
    ]
    for name in required_files:
        path = os.path.join(sim_data_path, name)
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            return False

    frames_dir = os.path.join(sim_data_path, "frames")
    if not os.path.isdir(frames_dir):
        return False
    if not any(fname.endswith(".png") for fname in os.listdir(frames_dir)):
        return False

    return True


def _default_vgen_prompt(*, motion_mode: str, motion_direction: Sequence[float], object_name: str) -> str:
    if motion_mode == "constant_velocity":
        return (
            f"The {object_name} is moving uniformly in direction {list(motion_direction)}. "
            "The camera stays still with realistic granular motion."
        )
    if motion_mode == "constant_acceleration":
        return (
            f"The {object_name} is accelerating in direction {list(motion_direction)}. "
            "The camera stays still with realistic granular motion."
        )
    if motion_mode == "free_fall":
        return f"The {object_name} falls down freely to the ground. The camera stays still with realistic granular motion."
    return ""


def _build_sim_config(
    *,
    case_name: str,
    case_output_dir: str,
    result_root: str,
    seed: int,
    object_name: str,
    img_prompt: str,
    vgen_prompt: str,
    motion_mode: str,
    motion_direction: Sequence[float],
    force_direction: Optional[Sequence[float]] = None,
    force_strength: Optional[float] = None,
    gravity: Optional[float] = None,
    initial_velocity: Optional[Sequence[float]] = None,
    simulated_frames_num: int = 51,
    estimated_real_size: Optional[float] = None,
) -> Dict[str, object]:
    if not vgen_prompt:
        vgen_prompt = _default_vgen_prompt(
            motion_mode=motion_mode,
            motion_direction=motion_direction,
            object_name=object_name,
        )

    config: Dict[str, object] = {
        "device": "cuda",
        "seed": seed,
        "example_name": case_name,
        "output_folder": os.path.join(result_root, case_name),
        "data_path": case_output_dir,
        "segmenter": "sam3",
        "all_object_names": [object_name],
        "sam3_mask_strategy": "best",
        "sam3d_use_raw_sam3_mask": True,
        "obj_kp_matching": True,
        "logging_level": "details",
        "debug": True,
        "stitched_inpainting": False,
        "mesh_resize_factor": 1.0,
        "target_faces": 10000,
        "dt": 0.02,
        "substeps": 10,
        "simulated_frames_num": simulated_frames_num,
        "frame_steps": 1,
        "material_type": ["rigid"],
        "use_primitive": True,
        "remap_depth": [1.0, 2.0],
        "rigid_rho": 1000,
        "rigid_friction": 0.01,
        "plane_friction": 0.01,
        "gravity": -1,
        "alpha_threshold": 0.8,
        "crop_start": 200,
        "fg_points_render_radius": 0.01,
        "num_output_frames": simulated_frames_num,
        "denoising_step_list": [800, 500, 250],
        "mask_dropin_step": 1,
        "franka_step": 0,
        "vgen_prompt": vgen_prompt,
        "output_width": 832,
        "output_height": 480,
        "sim_render_width": 832,
        "sim_render_height": 480,
        "render_bg_pointcloud": True,
        "preserve_aspect_ratio": True,
        "image_prompt": img_prompt,
        "motion_mode": motion_mode,
        "motion_direction": list(motion_direction),
        "estimated_real_size": estimated_real_size,
    }

    dir_x, dir_y, dir_z = motion_direction
    if motion_mode == "constant_velocity":
        speed = 2.0
        config["initial_velocity"] = [dir_x * speed, dir_y * speed, dir_z * speed, 0.0, 0.0, 0.0]
        config["force_direction"] = [0.0, 0.0, 0.0]
        config["force_strength"] = 0.0
    elif motion_mode == "constant_acceleration":
        config["initial_velocity"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        config["force_direction"] = [dir_x, dir_y, dir_z]
        config["force_strength"] = 10.0
    elif motion_mode == "free_fall":
        config["initial_velocity"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        config["gravity"] = -9.8
        config["force_direction"] = [0.0, 0.0, 0.0]
        config["force_strength"] = 0.0

    if force_direction is not None:
        config["force_direction"] = list(force_direction)
    if force_strength is not None:
        config["force_strength"] = float(force_strength)
    if gravity is not None:
        config["gravity"] = float(gravity)
    if initial_velocity is not None:
        config["initial_velocity"] = list(initial_velocity)

    return config


def _write_yaml(path: str, data: Dict[str, object]) -> None:
    import yaml

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _t2i_generate_images(
    *,
    model_name: str,
    jobs: List[Dict[str, object]],
    total: int,
    overwrite: bool,
    base_url: Optional[str] = None,
    fallback_root: str = "",
) -> None:
    import base64
    from io import BytesIO
    from PIL import Image
    from openai import OpenAI

    pending_jobs: List[Dict[str, object]] = []
    for job in jobs:
        out_dir = job["case_output_dir"]
        input_png = os.path.join(out_dir, "input.png")
        if (not overwrite) and os.path.exists(input_png) and os.path.getsize(input_png) > 0:
            print(f"[{job['idx']}/{total}] t2i case={job['case_name']} (input.png exists, skip)")
            continue
        pending_jobs.append(job)

    if not pending_jobs:
        print("All t2i targets already exist; skip vLLM request.")
        return

    resolved_base_url = (base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1").strip()
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    print(f"Connecting to vLLM API server for {len(pending_jobs)} cases... base_url={resolved_base_url}")
    client = OpenAI(
        api_key=api_key,
        base_url=resolved_base_url,
    )

    positive_magic = ", Ultra HD, 4K, cinematic composition."

    for job in pending_jobs:
        out_dir = job["case_output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        input_png = os.path.join(out_dir, "input.png")
        input_orig = os.path.join(out_dir, "input_original.png")
        print(f"[{job['idx']}/{total}] t2i case={job['case_name']} via vLLM")

        try:
            response = client.images.generate(
                model=model_name,
                prompt=job["image_prompt"] + positive_magic,
                n=1,
                size="832x480",
                extra_body={
                    "num_inference_steps": 50,
                    "true_cfg_scale": 4.0,
                    "seed": int(job["seed"])
                }
            )

            image_b64 = response.data[0].b64_json
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))

            image.save(input_orig)
            image.save(input_png)
        except Exception as e:
            print(f"Error generating image for {job['case_name']}: {e}")
            if fallback_root:
                fallback_img = os.path.join(fallback_root, str(job["case_name"]), "input.png")
                if os.path.exists(fallback_img) and os.path.getsize(fallback_img) > 0:
                    print(f"[{job['idx']}/{total}] t2i case={job['case_name']} fallback -> {fallback_img}")
                    shutil.copy2(fallback_img, input_orig)
                    shutil.copy2(fallback_img, input_png)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline from CSV, split into 3 logical stages:\n"
            "  t2i: text-to-image (generate input.png)\n"
            "  i2s: image-to-scene (write config.yaml)\n"
            "  s2v: scene-to-video (simulation + inference)\n\n"
            "Use --stages to run only part(s) of the pipeline so different conda envs can be used."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/lff/data1/cym/physical_data/RealWonder/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt",
        help="Path to inference checkpoint",
    )
    parser.add_argument("--output_root", type=str, default="cases", help="Default case output root")
    parser.add_argument("--result_root", type=str, default="result", help="Default result root")
    parser.add_argument("--sim_output_dir", type=str, default="000", help="Simulation output sub-folder")
    parser.add_argument("--seed", type=int, default=42, help="Default seed")
    parser.add_argument("--local_attn_size", type=int, default=21, help="Inference local attention size")
    parser.add_argument("--eval_degradation", type=float, default=0.05, help="Inference noise degradation")
    parser.add_argument("--franka_step", type=int, default=0, help="Initial anchoring index")
    parser.add_argument("--mask_dropin_step", type=int, default=1, help="Mask injection step")
    parser.add_argument("--denoising_step_list", type=str, default="[900, 600, 300, 100]", help="Denoising schedule list")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA generator weights")
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
        default="",
        help="Optional fallback root containing <case_name>/input.png when t2i endpoint is unavailable",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated stages: t2i,i2s,s2v (or 'all'). Default runs all unless --skip_* is used.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print planned work without running models.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (otherwise skip completed cases)")

    # t2i (text-to-image) memory controls
    parser.add_argument("--t2i_device", choices=["auto", "cuda", "cpu"], default="auto", help="t2i device selection")
    parser.add_argument("--t2i_dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="t2i dtype (CUDA only; CPU uses fp32)")
    parser.add_argument("--t2i_offload", choices=["none", "cpu", "sequential"], default="sequential", help="t2i offload mode to reduce VRAM usage")
    parser.add_argument("--t2i_device_map", choices=["none", "auto", "balanced", "balanced_low_0"], default="balanced", help="When --t2i_offload=none, optionally shard model across multiple GPUs")
    parser.add_argument("--t2i_max_memory_gb", type=int, default=46, help="Per-GPU max_memory (GiB) for --t2i_device_map sharding")

    # Back-compat flags.
    parser.add_argument("--skip_scene", action="store_true", help="(Deprecated) Skip t2i+i2s")
    parser.add_argument("--skip_simulation", action="store_true", help="(Deprecated) Skip physics simulation within s2v")
    parser.add_argument("--skip_inference", action="store_true", help="(Deprecated) Skip inference within s2v")

    args, additional_args = parser.parse_known_args()

    stages = _normalize_stage_list(args.stages)
    if not stages:
        stages = ["t2i", "i2s", "s2v"]
        if args.skip_scene:
            stages = [s for s in stages if s not in {"t2i", "i2s"}]
        if args.skip_simulation and args.skip_inference:
            stages = [s for s in stages if s != "s2v"]
    _validate_stages(stages)

    rows = load_jobs(args.csv_path)
    jobs: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        job = _build_job(row=row, idx=idx, args=args)
        if "t2i" in stages and not job["image_prompt"]:
            raise ValueError(f"Row {idx}: image_prompt is required when stage t2i is enabled")
        jobs.append(job)

    total = len(jobs)

    for stage_idx, stage in enumerate(stages, start=1):
        if stage == "t2i":
            print(f"Stage {stage_idx}/{len(stages)} t2i (text-to-image): {total} cases")
            if args.dry_run:
                for job in jobs:
                    print(f"[{job['idx']}/{total}] t2i case={job['case_name']} -> {os.path.join(job['case_output_dir'], 'input.png')}")
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

                config_path = job["config_path"]
                if (not args.overwrite) and os.path.exists(config_path) and os.path.getsize(config_path) > 0:
                    print(f"[{job['idx']}/{total}] i2s case={job['case_name']} (config.yaml exists, skip)")
                    continue

                input_path = os.path.join(job["case_output_dir"], "input.png")
                if not args.dry_run and not os.path.exists(input_path):
                    raise FileNotFoundError(
                        f"Missing input image for i2s: {input_path}. "
                        "Run with --stages t2i,i2s first (in imagegen env), or provide the image."
                    )

                sim_config = _build_sim_config(
                    case_name=job["case_name"],
                    case_output_dir=job["case_output_dir"],
                    result_root=args.result_root,
                    seed=int(job["seed"]),
                    object_name=job["object_name"],
                    img_prompt=job["image_prompt"],
                    vgen_prompt=job["vgen_prompt"],
                    motion_mode=job["motion_mode"],
                    motion_direction=job["motion_direction"],
                    force_direction=job["force_direction"],
                    force_strength=job["force_strength"],
                    gravity=job["gravity"],
                    initial_velocity=job["initial_velocity"],
                )

                if args.dry_run:
                    continue

                _write_yaml(job["config_path"], sim_config)

        elif stage == "s2v":
            print(f"Stage {stage_idx}/{len(stages)} s2v (scene-to-video): {total} cases")
            if args.dry_run:
                for job in jobs:
                    print(f"[{job['idx']}/{total}] s2v case={job['case_name']} (dry_run)")
                continue

            import subprocess
            import sys

            for job in jobs:
                if (not args.overwrite) and os.path.exists(job["output_path"]) and os.path.getsize(job["output_path"]) > 0:
                    print(f"[{job['idx']}/{total}] s2v case={job['case_name']} (final.mp4 exists, skip)")
                    continue

                print(f"[{job['idx']}/{total}] s2v case={job['case_name']} (via standalone sub-process)")

                s2v_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "s2v_standalone.py"),
                    "--config_path", job["config_path"],
                    "--checkpoint_path", args.checkpoint_path,
                    "--sim_output_dir", job["sim_output_dir"],
                    "--output_path", job["output_path"],
                    "--eval_degradation", str(job["eval_degradation"]),
                    "--franka_step", str(job["franka_step"]),
                    "--mask_dropin_step", str(job["mask_dropin_step"]),
                    "--denoising_step_list", str(job["denoising_step_list"]),
                    "--seed", str(job["seed"]),
                    "--local_attn_size", str(args.local_attn_size),
                ]
                if args.use_ema:
                    s2v_cmd.append("--use_ema")

                try:
                    subprocess.run(s2v_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Standalone s2v failed for {job['case_name']} with exit code {e.returncode}. Continuing next case.")
        else:
            raise AssertionError(f"Unhandled stage: {stage}")

    print(f"All done. Processed {total} cases. stages={','.join(stages)} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
