"""Batch final video generation from RealWonder simulation outputs.

This script reuses the Wan-based SDEdit inference runner from `infer_sim.py`
and generates `final.mp4` from an existing `final_sim/` directory.

Supported inputs:
- `--sim_data_path`: generate one case
- `--csv_path`: batch from CSV rows
- `--sim_root`: recursively discover `*/final_sim` directories under a root

The Wan model is loaded once and reused for all cases in the process.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

from batch_pipeline_csv import _build_job, _is_complete_final_sim, load_jobs
from infer_sim import SimInferenceRunner


DEFAULT_CHECKPOINT_PATH = (
	"/home/lff/data1/cym/physical_data/rw_sim/ckpts/Realwonder-Distilled-AR-I2V-Flow/"
	"sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt"
)
DEFAULT_RESULT_ROOT = "/home/lff/bigdata1/cym/realwonder_simdata/result"
DEFAULT_SIM_DATA_PATH = "/home/lff/bigdata1/cym/realwonder_simdata/result/apple_tree_drop/0/final_sim"


def _is_complete_final_video(output_path: str) -> bool:
	return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _infer_case_name(sim_data_path: str) -> str:
	sim_path = Path(sim_data_path).resolve()
	if sim_path.name == "final_sim" and sim_path.parent.name:
		parent = sim_path.parent.parent
		if parent.name:
			return parent.name
		return sim_path.parent.name
	return sim_path.parent.name or sim_path.stem


def _job_from_sim_path(sim_data_path: str, output_path: Optional[str] = None) -> Dict[str, object]:
	sim_path = Path(sim_data_path).resolve()
	if output_path is None:
		output_path = str(sim_path.parent / "final.mp4")
	return {
		"idx": 1,
		"case_name": _infer_case_name(str(sim_path)),
		"sim_data_path": str(sim_path),
		"output_path": output_path,
		"eval_degradation": None,
		"seed": None,
		"use_ema": None,
	}


def _collect_jobs_from_csv(csv_path: str, args) -> List[Dict[str, object]]:
	rows = load_jobs(csv_path)
	jobs: List[Dict[str, object]] = []
	for idx, row in enumerate(rows, start=1):
		job = _build_job(row=row, idx=idx, args=args)
		jobs.append(job)
	return jobs


def _collect_jobs_from_root(sim_root: str) -> List[Dict[str, object]]:
	root = Path(sim_root).resolve()
	if not root.exists():
		raise FileNotFoundError(f"Simulation root not found: {sim_root}")

	jobs: List[Dict[str, object]] = []
	for sim_dir in sorted(root.rglob("final_sim")):
		if not sim_dir.is_dir():
			continue
		jobs.append(
			{
				"idx": len(jobs) + 1,
				"case_name": _infer_case_name(str(sim_dir)),
				"sim_data_path": str(sim_dir),
				"output_path": str(sim_dir.parent / "final.mp4"),
				"eval_degradation": None,
				"seed": None,
				"use_ema": None,
			}
		)

	if not jobs:
		raise ValueError(f"No final_sim directories found under: {sim_root}")

	return jobs


def _normalize_jobs(jobs: List[Dict[str, object]], args) -> List[Dict[str, object]]:
	normalized: List[Dict[str, object]] = []
	for idx, job in enumerate(jobs, start=1):
		sim_data_path = str(job["sim_data_path"])
		output_path = str(job["output_path"])
		case_name = str(job.get("case_name") or _infer_case_name(sim_data_path))
		eval_degradation = job.get("eval_degradation")
		seed = job.get("seed")
		use_ema = job.get("use_ema")

		normalized.append(
			{
				"idx": idx,
				"case_name": case_name,
				"sim_data_path": sim_data_path,
				"output_path": output_path,
				"eval_degradation": args.eval_degradation if eval_degradation in (None, "") else float(eval_degradation),
				"seed": args.seed if seed in (None, "") else int(seed),
				"use_ema": args.use_ema if use_ema in (None, "") else bool(use_ema),
			}
		)

	return normalized


def _resolve_jobs(args) -> List[Dict[str, object]]:
	if args.csv_path:
		jobs = _collect_jobs_from_csv(args.csv_path, args)
		return _normalize_jobs(jobs, args)

	if args.sim_data_path:
		return _normalize_jobs([_job_from_sim_path(args.sim_data_path, args.output_path or None)], args)

	if args.sim_root:
		jobs = _collect_jobs_from_root(args.sim_root)
		return _normalize_jobs(jobs, args)

	return _normalize_jobs([_job_from_sim_path(DEFAULT_SIM_DATA_PATH, None)], args)


def _ensure_wan_model_root(model_root: str) -> None:
	root = Path(model_root).expanduser().resolve()
	if not root.exists():
		raise FileNotFoundError(f"WAN model root not found: {model_root}")

	alias = Path("wan_models")
	if alias.exists():
		return

	alias.symlink_to(root, target_is_directory=True)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Generate final.mp4 from RealWonder simulation outputs using the Wan model.\n\n"
			"Examples:\n"
			"  python batch_finalvideo.py --sim_data_path <case>/final_sim\n"
			"  python batch_finalvideo.py --csv_path cases.csv\n"
			"  python batch_finalvideo.py --sim_root /path/to/result\n"
		),
		formatter_class=argparse.RawTextHelpFormatter,
	)

	parser.add_argument("--csv_path", type=str, default="", help="CSV path for batch inference")
	parser.add_argument(
		"--sim_data_path",
		type=str,
		default="",
		help="Single-case final_sim directory (contains noises.npy, frames/, masks, config.yaml, etc.)",
	)
	parser.add_argument(
		"--sim_root",
		type=str,
		default="",
		help="Recursively discover final_sim directories under this root",
	)
	parser.add_argument(
		"--result_root",
		type=str,
		default=DEFAULT_RESULT_ROOT,
		help="Default result root for CSV-derived paths",
	)
	parser.add_argument(
		"--output_path",
		type=str,
		default="",
		help="Single-case output path (default: <sim_data_path>/../final.mp4)",
	)
	parser.add_argument(
		"--checkpoint_path",
		type=str,
		default=DEFAULT_CHECKPOINT_PATH,
		help="Path to the Wan checkpoint file",
	)
	parser.add_argument("--use_ema", action="store_true", help="Use EMA parameters from the checkpoint")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument(
		"--eval_degradation",
		type=float,
		default=0.5,
		help="Noise degradation level used during simulation-guided inference",
	)
	parser.add_argument("--local_attn_size", type=int, default=21, help="Local attention size for the Wan model")
	parser.add_argument(
		"--wan_model_root",
		type=str,
		default=os.environ.get("WAN_MODEL_ROOT", ""),
		help="Directory to symlink as ./wan_models before loading Wan checkpoints",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final.mp4 outputs")
	parser.add_argument("--dry_run", action="store_true", help="Validate inputs and print planned work only")

	args, additional_args = parser.parse_known_args()

	jobs = _resolve_jobs(args)
	total = len(jobs)

	os.chdir(Path(__file__).resolve().parent)
	if args.wan_model_root.strip():
		_ensure_wan_model_root(args.wan_model_root)

	print(f"Preparing Wan final video generation for {total} case(s)")
	runner = None if args.dry_run else SimInferenceRunner(
		checkpoint_path=args.checkpoint_path,
		use_ema=args.use_ema,
		seed=args.seed,
		local_attn_size=args.local_attn_size,
		additional_args=additional_args,
	)

	for job in jobs:
		sim_data_path = str(job["sim_data_path"])
		output_path = str(job["output_path"])
		case_name = str(job["case_name"])
		eval_degradation = float(job["eval_degradation"])

		if (not args.overwrite) and _is_complete_final_video(output_path):
			print(f"[{job['idx']}/{total}] final case={case_name} (final.mp4 exists, skip)")
			continue

		if not _is_complete_final_sim(sim_data_path):
			raise FileNotFoundError(
				f"Incomplete simulation artifacts for {case_name}: {sim_data_path}. "
				"Expected noises.npy, resized_input_image.png, prompt.txt, config.yaml and frames/*.png."
			)

		print(f"[{job['idx']}/{total}] final case={case_name}")
		print(f"  sim_data_path={sim_data_path}")
		print(f"  output_path={output_path}")
		print(f"  eval_degradation={eval_degradation}")

		if args.dry_run:
			continue

		output_dir = os.path.dirname(output_path)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)

		runner.infer_one(
			sim_data_path=sim_data_path,
			output_path=output_path,
			eval_degradation=eval_degradation,
		)

	print(f"All done. Processed {total} case(s). dry_run={args.dry_run}")


if __name__ == "__main__":
	main()