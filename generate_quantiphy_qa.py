import json
import csv
import math
import glob
import os
import argparse
import re

try:
    import yaml
except Exception:
    yaml = None


def _has_effective_prior(value, eps=1e-12):
    try:
        return abs(float(value)) > eps
    except Exception:
        return True

def load_sim_metadata(metadata_file):
    with open(metadata_file, 'r') as f:
        return json.load(f)


def _load_object_names_from_config(metadata_file):
    if yaml is None:
        return []

    sim_dir = os.path.dirname(metadata_file)
    candidate_paths = [
        os.path.join(sim_dir, 'config.yaml'),
        os.path.join(os.path.dirname(sim_dir), 'config.yaml'),
        os.path.join(os.path.dirname(sim_dir), 'final_sim', 'config.yaml'),
        os.path.join(os.path.dirname(os.path.dirname(sim_dir)), 'final_sim', 'config.yaml'),
    ]

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            names = cfg.get('all_object_names') or []
            if isinstance(names, list):
                return [str(x) for x in names if str(x).strip()]
        except Exception:
            continue
    return []


def _resolve_object_name(obj_idx_str, configured_names, preferred_primary_name=None):
    idx = None
    try:
        idx = int(obj_idx_str)
    except Exception:
        m = re.match(r"^object_(\d+)$", str(obj_idx_str).strip())
        if m:
            idx = int(m.group(1))

    if idx is None:
        return str(obj_idx_str)

    if 0 <= idx < len(configured_names):
        name = str(configured_names[idx]).strip()
        if name:
            return name

    if idx == 0 and preferred_primary_name:
        return str(preferred_primary_name)

    return f"object_{idx}"

import random

def _has_effective_prior(value, eps=1e-3):
    try:
        # Avoid 0.0 or near-zero values which make QA meaningless
        return abs(float(value)) > eps
    except Exception:
        return True

def _get_estimated_size_info(metadata_file):
    """Retrieve fixed estimated size and preferred unit from config."""
    if yaml is None:
        return None, 'm'
    sim_dir = os.path.dirname(metadata_file)
    candidate_paths = [
        os.path.join(sim_dir, 'config.yaml'),
        os.path.join(os.path.dirname(sim_dir), 'config.yaml'),
        os.path.join(os.path.dirname(sim_dir), 'final_sim', 'config.yaml'),
    ]
    for path in candidate_paths:
        if not os.path.exists(path): continue
        try:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            val = cfg.get('estimated_real_size') 
            if val is None:
                continue
            
            if isinstance(val, str):
                val_s = val.strip().lower()
                if val_s.endswith('cm'):
                    return float(val_s[:-2]) / 100.0, 'cm'
                elif val_s.endswith('m'):
                    return float(val_s[:-1]), 'm'
                else:
                    return float(val), 'm'
            return float(val), 'm'
        except: continue
    return None, 'm'


def _format_time_label(seconds, precision="auto"):
    seconds = float(seconds)
    if precision == "fixed2":
        return f"{seconds:.2f}s"
    if precision == "fixed1":
        return f"{seconds:.1f}s"
    if precision == "integer":
        return f"{int(round(seconds))}s"
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{int(round(seconds))}s"
    if abs(seconds * 10 - round(seconds * 10)) < 1e-6:
        return f"{seconds:.1f}s"
    return f"{seconds:.2f}s"


def _format_interval_label(start_s, end_s, precision="fixed2", mode="between"):
    start_label = _format_time_label(start_s, precision)
    end_label = _format_time_label(end_s, precision)
    if mode == "in":
        return f"in {start_label} to {end_label}"
    return f"between {start_label} and {end_label}"


def _compute_average_travel_speed(history, start_idx, end_idx, fps=24, scale=1.0):
    if end_idx <= start_idx or start_idx < 0 or end_idx >= len(history):
        return None

    total_distance = 0.0
    prev_pos = history[start_idx].get("position")
    if prev_pos is None:
        return None

    for idx in range(start_idx + 1, end_idx + 1):
        cur_pos = history[idx].get("position")
        if cur_pos is None:
            return None
        total_distance += math.sqrt(
            sum((float(a) - float(b)) ** 2 for a, b in zip(prev_pos, cur_pos))
        )
        prev_pos = cur_pos

    duration = float(end_idx - start_idx) / float(fps)
    if duration <= 0:
        return None
    return (total_distance * scale) / duration


def _pick_time_candidates(history_len, fps, preferred_times):
    max_time = max(0.0, (history_len - 1) / float(fps))
    return [t for t in preferred_times if t <= max_time + 1e-6]


def _add_qa(qas, seen_questions, qa):
    question = qa.get('question', '')
    key = (qa.get('inference_type', ''), question)
    if question and key in seen_questions:
        return
    seen_questions.add(key)
    qas.append(qa)


def _load_grayscale_mask(mask_path):
    if not os.path.exists(mask_path) or os.path.getsize(mask_path) <= 0:
        return None

    try:
        from PIL import Image
    except Exception:
        return None

    with Image.open(mask_path) as image:
        import numpy as np

        return np.asarray(image.convert('L'), dtype=np.uint8)


def _is_all_black_mask(mask):
    try:
        return mask.size == 0 or not mask.any()
    except Exception:
        return True


def _resolve_case_dir(meta_file):
    sim_dir = os.path.dirname(meta_file)
    case_dir = os.path.dirname(sim_dir)
    case_name = os.path.basename(case_dir)
    return case_name, case_dir


def _default_mask_only_skip_case_names(metadata_files):
    skip_case_names = []
    for meta_file in metadata_files:
        case_name, case_dir = _resolve_case_dir(meta_file)
        mask_path = os.path.join(case_dir, 'inpainter_masks.png')
        mask = _load_grayscale_mask(mask_path)
        if mask is None or _is_all_black_mask(mask):
            skip_case_names.append(case_name)
    return skip_case_names

def generate_qa_from_metadata(
    metadata,
    video_id,
    meta_full_path,
    object_names=None,
    preferred_primary_name=None,
    video_source=None,
):
    qas = []
    seen_questions = set()
    fps = 24
    objects = metadata.get('objects', {})
    configured_names = object_names or []
    resolved_video_source = video_source or ''
    
    estimated_real_size, preferred_unit = _get_estimated_size_info(meta_full_path)
    video_prior_type = random.choice(['length', 'speed', 'accel'])

    # Unit formatting based on input preference
    def format_val(val, base_unit='m', force_unit=None):
        target_unit = force_unit or base_unit
        if target_unit == 'cm' or (not force_unit and base_unit == 'm' and (val < 0.1 or (0.1 <= val < 1.0 and random.random() < 0.3))):
            return round(val * 100, 2), 'cm'
        return round(val, 3), 'm'

    def add_question(question, inference_type, prior, posterior, extra=None):
        qa = {
            'video_id': video_id,
            'video_source': resolved_video_source,
            'inference_type': inference_type,
            'question': question,
            'ground_truth_prior': prior,
            'ground_truth_posterior': posterior,
        }
        if extra:
            qa.update(extra)
        _add_qa(qas, seen_questions, qa)

    def choose_motion_word(motion_mode):
        if motion_mode == 'free_fall':
            return random.choice(['speed', 'velocity'])
        if motion_mode == 'constant_velocity':
            return random.choice(['speed', 'velocity'])
        return random.choice(['speed', 'velocity'])

    def point_time_candidates(history_len):
        return _pick_time_candidates(history_len, fps, [0.5, 1.0, 1.5, 2.0, 3.0])

    def interval_candidates(history_len):
        candidates = []
        if history_len > fps:
            candidates.extend([
                (0.0, 0.5),
                (0.5, 1.0),
                (1.0, 2.0),
            ])
        if history_len > 2 * fps:
            candidates.extend([
                (0.0, 1.0),
                (1.0, 3.0),
            ])
        if history_len > 3 * fps:
            candidates.append((0.0, 2.0))
        return [
            (start_s, end_s)
            for start_s, end_s in candidates
            if int(round(end_s * fps)) < history_len
        ]

    def add_length_question(obj_name, length_val, length_unit, prior_text):
        templates = [
            f"What is the length of the {obj_name} in {length_unit}?",
            f"How long is the {obj_name} in {length_unit}?",
        ]
        add_question(
            random.choice(templates),
            'SS',
            prior_text,
            length_val,
        )

    def add_point_motion_question(obj_name, motion_word, t_target, unit_label, prior_text, answer_value):
        time_fixed1 = _format_time_label(t_target, 'fixed1')
        time_auto = _format_time_label(t_target, 'auto')
        templates = [
            f"What is the {motion_word} of the {obj_name} at time {time_fixed1} in {unit_label}?",
            f"What is the {motion_word} of the {obj_name} at {time_fixed1} in {unit_label}?",
            f"What is the {motion_word} of the {obj_name} at {time_auto} in {unit_label}?",
        ]
        add_question(random.choice(templates), 'SD', prior_text, answer_value)

    def add_accel_question(obj_name, t_target, unit_label, prior_text, answer_value):
        time_fixed1 = _format_time_label(t_target, 'fixed1')
        templates = [
            f"What is the acceleration of the {obj_name} at time {time_fixed1} in {unit_label}?",
            f"What is the acceleration of the {obj_name} at {time_fixed1} in {unit_label}?",
        ]
        add_question(random.choice(templates), 'SD', prior_text, answer_value)

    def add_average_motion_question(obj_name, start_s, end_s, unit_label, prior_text, answer_value):
        interval_style = random.choice(['between', 'in'])
        interval_label = _format_interval_label(start_s, end_s, precision='fixed2', mode=interval_style)
        avg_word = random.choice(['average speed', 'average velocity'])
        templates = [
            f"What is the {avg_word} of the {obj_name} {interval_label} in {unit_label}?",
            f"What is the {obj_name}'s {avg_word} {interval_label} in {unit_label}?",
        ]
        add_question(random.choice(templates), 'SD', prior_text, answer_value)

    for obj_idx_str, obj_data in objects.items():
        obj_name = _resolve_object_name(obj_idx_str, configured_names, preferred_primary_name)
        
        sim_length = obj_data.get('size', [1.0])[0]
        scale = 1.0
        if estimated_real_size:
            scale = float(estimated_real_size) / sim_length

        length_val, length_unit = format_val(sim_length * scale, 'm', force_unit=preferred_unit)
        size_prior = f"length of the {obj_name} = {length_val} {length_unit}"
        has_size_prior = _has_effective_prior(length_val)

        if 'history' in obj_data:
            history = obj_data['history']
            motion_mode = str(obj_data.get('motion_mode', '')).strip()
            history_len = len(history)
            point_times = point_time_candidates(history_len)
            interval_times = interval_candidates(history_len)

            speed_samples = []
            accel_samples = []
            for t_target in point_times:
                frame_idx = min(int(round(t_target * fps)), history_len - 1)
                raw_speed = history[frame_idx].get("velocity_abs", 0.0)
                raw_accel = history[frame_idx].get("acceleration_abs", 0.0)

                speed_v, speed_u = format_val(raw_speed * scale, 'm', force_unit=preferred_unit)
                accel_v, accel_u = format_val(raw_accel * scale, 'm', force_unit=preferred_unit)

                speed_unit = f"{speed_u}/s"
                accel_unit = f"{accel_u}/s^2"
                if _has_effective_prior(speed_v):
                    speed_samples.append((t_target, speed_v, speed_unit))
                if _has_effective_prior(accel_v):
                    accel_samples.append((t_target, accel_v, accel_unit))

            selected_speed = random.choice(speed_samples) if speed_samples else None
            selected_accel = random.choice(accel_samples) if accel_samples else None
            selected_interval = random.choice(interval_times) if interval_times else None

            motion_word = choose_motion_word(motion_mode)
            effective_prior_type = video_prior_type
            if effective_prior_type == 'speed' and not speed_samples:
                effective_prior_type = 'accel' if accel_samples else 'length'
            elif effective_prior_type == 'accel' and not accel_samples:
                effective_prior_type = 'speed' if speed_samples else 'length'

            if effective_prior_type == 'length':
                if selected_speed is not None:
                    t_target, speed_v, speed_unit = selected_speed
                    add_point_motion_question(
                        obj_name,
                        random.choice(['speed', 'velocity']),
                        t_target,
                        speed_unit,
                        size_prior,
                        speed_v,
                    )

                if selected_accel is not None:
                    t_target, accel_v, accel_unit = selected_accel
                    add_accel_question(
                        obj_name,
                        t_target,
                        accel_unit,
                        size_prior,
                        accel_v,
                    )

                if selected_interval is not None:
                    start_s, end_s = selected_interval
                    start_idx = int(round(start_s * fps))
                    end_idx = int(round(end_s * fps))
                    avg_speed = _compute_average_travel_speed(history, start_idx, end_idx, fps=fps, scale=scale)
                    if avg_speed is None or not _has_effective_prior(avg_speed):
                        avg_speed = None
                    if avg_speed is not None:
                        avg_v, avg_u = format_val(avg_speed, 'm', force_unit=preferred_unit)
                        avg_unit = f"{avg_u}/s"
                        add_average_motion_question(
                            obj_name,
                            start_s,
                            end_s,
                            avg_unit,
                            size_prior,
                            avg_v,
                        )

            elif effective_prior_type == 'speed' and speed_samples:
                prior_t, prior_speed_v, prior_speed_unit = selected_speed or speed_samples[0]
                speed_prior = f"speed of the {obj_name} at {_format_time_label(prior_t, 'fixed1')} = {prior_speed_v} {prior_speed_unit}"
                if has_size_prior:
                    add_length_question(obj_name, length_val, length_unit, speed_prior)

                if selected_accel is not None:
                    t_target, accel_v, accel_unit = selected_accel
                    add_accel_question(
                        obj_name,
                        t_target,
                        accel_unit,
                        speed_prior,
                        accel_v,
                    )

                if selected_interval is not None:
                    start_s, end_s = selected_interval
                    start_idx = int(round(start_s * fps))
                    end_idx = int(round(end_s * fps))
                    avg_speed = _compute_average_travel_speed(history, start_idx, end_idx, fps=fps, scale=scale)
                    if avg_speed is None or not _has_effective_prior(avg_speed):
                        avg_speed = None
                    if avg_speed is not None:
                        avg_v, avg_u = format_val(avg_speed, 'm', force_unit=preferred_unit)
                        avg_unit = f"{avg_u}/s"
                        add_average_motion_question(
                            obj_name,
                            start_s,
                            end_s,
                            avg_unit,
                            speed_prior,
                            avg_v,
                        )

            elif effective_prior_type == 'accel' and accel_samples:
                prior_t, prior_accel_v, prior_accel_unit = selected_accel or accel_samples[0]
                accel_prior = f"acceleration of the {obj_name} at {_format_time_label(prior_t, 'fixed1')} = {prior_accel_v} {prior_accel_unit}"
                if has_size_prior:
                    add_length_question(obj_name, length_val, length_unit, accel_prior)

                if selected_speed is not None:
                    t_target, speed_v, speed_unit = selected_speed
                    add_point_motion_question(
                        obj_name,
                        random.choice(['speed', 'velocity']),
                        t_target,
                        speed_unit,
                        accel_prior,
                        speed_v,
                    )

                if selected_interval is not None:
                    start_s, end_s = selected_interval
                    start_idx = int(round(start_s * fps))
                    end_idx = int(round(end_s * fps))
                    avg_speed = _compute_average_travel_speed(history, start_idx, end_idx, fps=fps, scale=scale)
                    if avg_speed is None or not _has_effective_prior(avg_speed):
                        avg_speed = None
                    if avg_speed is not None:
                        avg_v, avg_u = format_val(avg_speed, 'm', force_unit=preferred_unit)
                        avg_unit = f"{avg_u}/s"
                        add_average_motion_question(
                            obj_name,
                            start_s,
                            end_s,
                            avg_unit,
                            accel_prior,
                            avg_v,
                        )


    # Distance QA between first two objects at t=0.5s
    obj_keys = list(objects.keys())
    if len(obj_keys) >= 2:
        obj1, obj2 = obj_keys[0], obj_keys[1]
        obj1_name = _resolve_object_name(
            obj1,
            configured_names,
            preferred_primary_name=preferred_primary_name,
        )
        obj2_name = _resolve_object_name(
            obj2,
            configured_names,
            preferred_primary_name=preferred_primary_name,
        )
        t_target = 0.5
        frame_idx = int(t_target * fps)
        
        history1 = objects[obj1].get('history', [])
        history2 = objects[obj2].get('history', [])
        
        if frame_idx < len(history1) and frame_idx < len(history2):
            p1 = history1[frame_idx].get("position", [0,0,0])
            p2 = history2[frame_idx].get("position", [0,0,0])
            sim_dist = math.sqrt(sum((a - b)**2 for a, b in zip(p1, p2)))
            
            # Use format_val to maintain unit consistency
            dist_val, dist_unit = format_val(sim_dist * scale, 'm', force_unit=preferred_unit)

            question_text = random.choice([
                f'What is the distance between the {obj1_name} and the {obj2_name} at {t_target}s in {dist_unit}?',
                f'What is the distance between {obj1_name} and {obj2_name} at {t_target}s in {dist_unit}?',
                f'What is the distance from {obj1_name} to {obj2_name} at {t_target}s in {dist_unit}?',
            ])

            if video_prior_type == 'length' and has_size_prior:
                l_v, l_u = format_val(objects[obj1]['size'][0] * scale, 'm', force_unit=preferred_unit)
                prior1 = f"length of the {obj1_name} = {l_v} {l_u}"
                if _has_effective_prior(l_v):
                    _add_qa(qas, seen_questions, {
                        'video_id': video_id,
                        'video_source': resolved_video_source,
                        'video_type': 'S2MX',
                        'fps': fps,
                        'inference_type': 'SS',
                        'question': question_text,
                        'ground_truth_prior': prior1,
                        'depth_info': '',
                        'ground_truth_posterior': dist_val,
                    })

            elif video_prior_type == 'speed' and frame_idx < len(history1):
                raw_s1 = history1[frame_idx].get("velocity_abs", 0.0)
                s_v, s_u = format_val(raw_s1 * scale, 'm', force_unit=preferred_unit)
                s_unit = f"{s_u}/s"
                prior2 = f"speed of the {obj1_name} at {t_target}s = {s_v} {s_unit}"
                if _has_effective_prior(s_v):
                    _add_qa(qas, seen_questions, {
                        'video_id': video_id,
                        'video_source': resolved_video_source,
                        'video_type': 'V2MX',
                        'fps': fps,
                        'inference_type': 'DS',
                        'question': question_text,
                        'ground_truth_prior': prior2,
                        'depth_info': '',
                        'ground_truth_posterior': dist_val,
                    })

            elif video_prior_type == 'accel' and frame_idx < len(history1):
                raw_a1 = history1[frame_idx].get("acceleration_abs", 0.0)
                a_v, a_u = format_val(raw_a1 * scale, 'm', force_unit=preferred_unit)
                a_unit = f"{a_u}/s^2"
                prior3 = f"acceleration of the {obj1_name} at {t_target}s = {a_v} {a_unit}"
                if _has_effective_prior(a_v):
                    _add_qa(qas, seen_questions, {
                        'video_id': video_id,
                        'video_source': resolved_video_source,
                        'video_type': 'V2MX',
                        'fps': fps,
                        'inference_type': 'DS',
                        'question': question_text,
                        'ground_truth_prior': prior3,
                        'depth_info': '',
                        'ground_truth_posterior': dist_val,
                    })
            
    return qas

def generate_csv(input_dir, output_csv, preferred_primary_name=None, skip_case_names=None, qa_filter_strategy='mask_only'):
    all_qas = []
    metadata_files = glob.glob(os.path.join(input_dir, '**/kinematics_log.json'), recursive=True)
    normalized_strategy = (qa_filter_strategy or 'mask_only').strip().lower()
    if skip_case_names is None:
        if normalized_strategy == 'mask_only':
            skip_case_names = _default_mask_only_skip_case_names(metadata_files)

    skip_case_names = {
        str(name).strip()
        for name in (skip_case_names or [])
        if str(name).strip()
    }
    
    for i, meta_file in enumerate(metadata_files):
        video_id = f"sim_{i:04d}"
        metadata = load_sim_metadata(meta_file)
        object_names = _load_object_names_from_config(meta_file)

        auto_case_name = ""
        try:
            # .../<result_root>/<case_name>/<sim_output_dir>/simulation/kinematics_log.json
            auto_case_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(meta_file))))
        except Exception:
            auto_case_name = ""

        resolved_primary_name = preferred_primary_name or auto_case_name or None
        if auto_case_name in skip_case_names:
            print(f"Skipping QA for {auto_case_name} due to quality filter")
            continue

        case_dir = os.path.dirname(os.path.dirname(meta_file))
        video_source = os.path.join(case_dir, 'final_sim', 'simulation.mp4')

        qas = generate_qa_from_metadata(
            metadata,
            video_id,
            meta_full_path=meta_file,
            object_names=object_names,
            preferred_primary_name=resolved_primary_name,
            video_source=video_source,
        )
        all_qas.extend(qas)
        
    fieldnames = ['video_id', 'video_source', 'video_type', 'fps', 'inference_type', 'question', 'ground_truth_prior', 'depth_info', 'ground_truth_posterior']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_qas)
    
    print(f"Generated {len(all_qas)} QA pairs and saved to {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate QA pairs from simulation kinematics logs')
    parser.add_argument('--input_dir', type=str, default='result/', help='Root directory to scan kinematics logs')
    parser.add_argument('--output_csv', type=str, default='quantiphy_synthetic_dataset.csv', help='Output CSV path')
    parser.add_argument('--preferred_primary_name', type=str, default='', help='Optional fallback object name for index 0')
    parser.add_argument(
        '--qa_filter_strategy',
        type=str,
        default='mask_only',
        choices=['mask_only', 'none'],
        help='QA filtering strategy (default: mask_only)',
    )
    args = parser.parse_args()

    generate_csv(
        args.input_dir,
        args.output_csv,
        preferred_primary_name=args.preferred_primary_name or None,
        qa_filter_strategy=args.qa_filter_strategy,
    )
