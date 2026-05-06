from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import torch
import sys
import os
from contextlib import nullcontext
import cv2
from repvit_sam import SamAutomaticMaskGenerator, sam_model_registry
import urllib.request
import PIL
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sam2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2ImagePredictor

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w).astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

import os
def show_masks(image, masks, scores, save_prefix, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        os.makedirs("debug/sam2", exist_ok=True)
        plt.savefig(f"debug/sam2/{save_prefix}_masks_{i:02d}.png")
        plt.close(fig)


class OneFormerSegmenter:
    _CACHE = {}

    def __init__(self, device="cuda"):
        self.device = device
        self.segment_processor = None
        self.segment_model = None
        self.load_model()
        
    def load_model(self):
        """Load the OneFormer model and processor"""
        cache_key = ("shi-labs/oneformer_ade20k_swin_large", str(self.device))
        if cache_key not in self._CACHE:
            segment_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large"
            )
            segment_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large"
            ).to(self.device)
            self._CACHE[cache_key] = (segment_processor, segment_model)
            print("OneFormer model loaded successfully")
        else:
            print("OneFormer model reused from cache")
        self.segment_processor, self.segment_model = self._CACHE[cache_key]
        
    def __call__(self, image, target_class_names):
        """Run semantic segmentation on the given image"""
        if self.segment_processor is None or self.segment_model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Check if input_image is a tensor and convert to PIL Image if needed
        if torch.is_tensor(image):
            # Ensure tensor is in correct format [B, C, H, W] or [C, H, W]
            if image.dim() == 4:
                image = image.squeeze(0)  # Remove batch dimension if present
            
            # Convert tensor to PIL Image using ToPILImage()
            if image.dim() == 3:
                image = ToPILImage()(image)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {image.shape}")

        segmenter_input = self.segment_processor(
            image, ["semantic"], return_tensors="pt"
        )
        segmenter_input = {
            name: tensor.to(self.device) for name, tensor in segmenter_input.items()
        }
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
            segment_output, target_sizes=[image.size[::-1]]
        )[0]
        
        id2label = self.segment_model.config.id2label
        label2id = {v.lower(): k for k, v in id2label.items()}
        
        target_masks = []
        for name in target_class_names:
            name_lower = name.lower()
            matched_id = None
            for label, obj_id in label2id.items():
                # Allow partial matches, e.g. "lamp" matches "sconce, sconce, lamp,..."
                if name_lower in label:
                    matched_id = obj_id
                    break
            
            if matched_id is not None:
                mask = (pred_semantic_map == matched_id).cpu().numpy()
                target_masks.append(mask)
            else:
                print(f"Warning: could not find class {name} in OneFormer labels.")
                target_masks.append(np.zeros(pred_semantic_map.shape, dtype=bool).cpu().numpy())

        return target_masks
    
class RepViTSegmenter:
    _CACHE = {}

    def __init__(self, device="cuda"):
        self.device = device
        self.repvit_segmenter = None
        self.load_model()

    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(current_dir, "repvit_sam.pt")
        
        if not os.path.exists(ckpt_path):
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            print(f"Downloading RepViT-SAM checkpoint to {ckpt_path}...")
            urllib.request.urlretrieve(
                "https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt",
                ckpt_path
            )
            print("Checkpoint downloaded successfully")
        cache_key = (ckpt_path, str(self.device))
        if cache_key not in self._CACHE:
            model_type = "repvit"
            repvit_sam = sam_model_registry[model_type](checkpoint=ckpt_path)
            repvit_sam = repvit_sam.to(self.device)
            repvit_sam.eval()

            repvit_segmenter = SamAutomaticMaskGenerator(
                model=repvit_sam,
                points_per_side=16,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.9,
                # min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
            self._CACHE[cache_key] = repvit_segmenter
            print("RepViT-SAM model loaded successfully")
        else:
            print("RepViT-SAM model reused from cache")

        self.repvit_segmenter = self._CACHE[cache_key]

    def __call__(self, image, target_class=[0], merge_mask=False):
        """
        Run RepViT-SAM segmentation to generate instance masks.
        
        Args:
            image: Input image as tensor [B, C, H, W] or [C, H, W] or PIL Image
            target_class: Unused parameter for compatibility
            
        Returns:
            List of dictionaries, each containing:
            - 'segmentation': boolean numpy array of shape (H, W) indicating mask
            - 'area': int, number of pixels in the mask
            - 'bbox': list [x, y, w, h] bounding box coordinates
            - 'predicted_iou': float, predicted IoU score
            - 'point_coords': list of [x, y] coordinates used for prediction
            - 'stability_score': float, stability score of the mask
            - 'crop_box': list [x0, y0, x1, y1] crop box used for prediction
        """
        assert isinstance(image, PIL.Image.Image), f"Image must be a PIL Image, but got {type(image)}"
        image_np = np.array(image)

        output = self.repvit_segmenter.generate(image_np)

        # for debug
        sam_masks_np = []
        for sid, sam_mask in enumerate(output):
            if sam_mask['area'] < 100:
                continue
            sam_masks_np.append(sam_mask['segmentation'])   # (512, 512) bool numpy array
            sam_mask = sam_mask['segmentation'] * 255
            sam_mask = sam_mask.astype(np.uint8)
            cv2.imwrite(f"debug/sam/sam_mask_{sid:02d}.png", sam_mask)
            cv2.imwrite(f"debug/sam/sam_mask_{sid:02d}_rgb.png", (sam_mask[:,:,None]/255).astype(np.uint8) * image_np[:,:,[2,1,0]])
        
        # # Dilate the mask(s) using cv2.dilate
        # kernel = np.ones((5, 5), np.uint8)  # You can adjust the kernel size as needed
        # dilated_sam_masks = []
        # for sid, sam_mask in enumerate(sam_masks_np):
        #     dilated = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=1)
        #     dilated_bool = dilated.astype(bool)
        #     dilated_sam_masks.append(dilated_bool)
        #     # Save dilated mask for debug if needed
        #     cv2.imwrite(f"debug/sam_dilated/sam_mask_{sid:02d}_dilated.png", dilated * 255)
        # # Optionally, you may want to replace sam_masks_np with dilated_sam_masks for later processing:
        # sam_masks_np = dilated_sam_masks

        # Dilate the mask(s) before returning
        # kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
        target_masks_np = []
        for target_id in target_class:
            if merge_mask:
                # If merge_mask is True, combine all masks corresponding to any part_id in target_id
                merged_mask = np.zeros_like(output[0]['segmentation'], dtype=bool)
                for part_id in target_id:
                    if part_id >= 0:
                        merged_mask = np.logical_or(merged_mask, output[part_id]['segmentation'])
                    else:
                        real_part_id = -(part_id + 1)
                        merged_mask = np.logical_or(merged_mask, np.logical_not(output[real_part_id]['segmentation']))
                # Dilate merged_mask
                # dilated_mask = cv2.dilate(merged_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # target_masks_np.append(dilated_mask)
                target_masks_np.append(merged_mask)
            else:
                # If merge_mask is False, take mask corresponding to target_id (single index)
                if target_id >= 0:
                    mask = output[target_id]['segmentation']
                else:
                    real_part_id = -(target_id + 1)
                    mask = np.logical_not(output[real_part_id]['segmentation'])
                # dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                # target_masks_np.append(dilated_mask)
                target_masks_np.append(mask)

        # import pdb; pdb.set_trace()

        # Optionally, for debugging, you can save the final dilated masks:
        # for tid, dilated_mask in enumerate(target_masks_np):
        #     cv2.imwrite(f"debug/sam_dilated/final_mask_{tid:02d}_dilated.png", dilated_mask.astype(np.uint8) * 255)
        
        return target_masks_np


class SegmentAnythingSegmenter:
    _MODEL_CACHE = {}

    def __init__(self, config, device="cuda"):
        self.device = device
        self.sam2_checkpoint = "/home/lff/data1/cym/physical_data/RealWonder/submodules/sam2/checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.config = config
        self.sam2_model = self._get_or_load_model()

    def _get_or_load_model(self):
        cache_key = (self.model_cfg, self.sam2_checkpoint, str(self.device))
        if cache_key not in self._MODEL_CACHE:
            self._MODEL_CACHE[cache_key] = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
            print("SAM2 model loaded successfully")
        else:
            print("SAM2 model reused from cache")
        return self._MODEL_CACHE[cache_key]

    def __call__(self, image):
        image = np.array(image)
        predictor = SAM2ImagePredictor(self.sam2_model)
        predictor.set_image(image)

        all_object_points = self.config['all_object_points']
        all_object_masks_idx = self.config['all_object_masks_idx']

        output_masks = []
        for object_idx, object_points in enumerate(all_object_points):
            # for each object
            object_points = np.array(object_points)
            object_points_xy = object_points[:, :2].copy()
            object_point_labels = object_points[:, 2].copy()

            fig = plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_points(object_points_xy, object_point_labels, plt.gca())
            plt.axis('on')
            plt.savefig(f"debug/sam2/input_points_{object_idx:02d}.png")
            plt.close(fig)
        
            masks, scores, logits = predictor.predict(
                point_coords=object_points_xy,
                point_labels=object_point_labels,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            show_masks(image, masks, scores, f"object_{object_idx:02d}", point_coords=object_points_xy, input_labels=object_point_labels, borders=True)
            output_masks.append(masks[all_object_masks_idx[object_idx]])
        
        return output_masks



class SegmentAnything3Segmenter:
    _MODEL_CACHE = {}
    _PROCESSOR_CACHE = {}

    def __init__(self, config, device="cuda"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.device = device
        self.config = config
        checkpoint_path = '/home/lff/bigdata1/huggingface/SAM3/sam3.pt'
        model_key = (checkpoint_path, str(device))
        if model_key not in self._MODEL_CACHE:
            self._MODEL_CACHE[model_key] = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=False
            ).to(device)
            print("SAM3 model loaded successfully")
        else:
            print("SAM3 model reused from cache")
        self.model = self._MODEL_CACHE[model_key]

        if model_key not in self._PROCESSOR_CACHE:
            self._PROCESSOR_CACHE[model_key] = Sam3Processor(self.model, device=device)
        self.processor = self._PROCESSOR_CACHE[model_key]
        self._configure_processor_precision()

    def _configure_processor_precision(self):
        # SAM3 upstream processor enters a persistent bf16 autocast context by default.
        # On some stacks this triggers vitdet linear bf16/float mismatch; use fp16 by default.
        precision = str(os.environ.get("REALWONDER_SAM3_PRECISION", "fp16")).strip().lower()
        if not str(self.device).startswith("cuda"):
            return

        old_ctx = getattr(self.processor, "bf16_context", None)
        if old_ctx is not None:
            try:
                old_ctx.__exit__(None, None, None)
            except Exception:
                pass

        if precision in {"none", "fp32", "float32", "off"}:
            new_ctx = nullcontext()
        elif precision in {"bf16", "bfloat16"}:
            new_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            # default: fp16
            new_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)

        try:
            new_ctx.__enter__()
        except Exception:
            # Fallback to no autocast if the selected dtype is unavailable.
            new_ctx = nullcontext()
            new_ctx.__enter__()

        self.processor.bf16_context = new_ctx

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            import PIL.Image as PILImage
            image = PILImage.fromarray(image)
        elif torch.is_tensor(image):
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.dim() == 3:
                image = ToPILImage()(image)

        inference_state = self.processor.set_image(image)
        image_h, image_w = image.height, image.width
        
        all_object_names = self.config['all_object_names']
        mask_strategy = self.config.get('sam3_mask_strategy', 'best')
        score_thresh = float(self.config.get('sam3_mask_score_thresh', 0.0))

        output_masks = []
        for object_idx, object_name in enumerate(all_object_names):
            # Prompt the model with text
            out_state = self.processor.set_text_prompt(state=inference_state, prompt=object_name)
            
            masks = out_state.get("masks", None)
            scores = out_state.get("scores", None)
            masks_logits = out_state.get("masks_logits", None)
            
            if masks is not None and len(masks) > 0:
                masks_t = masks.detach().float()
                if masks_t.ndim == 4 and masks_t.shape[1] == 1:
                    masks_t = masks_t[:, 0]

                # Prefer logits for size-correct postprocess to avoid direct resize on binary masks.
                logits_t = None
                if masks_logits is not None:
                    logits_t = masks_logits.detach().float()
                    if logits_t.ndim == 4 and logits_t.shape[1] == 1:
                        logits_t = logits_t[:, 0]

                scores_np = scores.detach().float().cpu().numpy() if scores is not None else None

                if scores_np is not None and score_thresh > 0:
                    keep_idx = np.where(scores_np >= score_thresh)[0]
                    if keep_idx.size > 0:
                        keep_idx_t = torch.as_tensor(keep_idx, device=masks_t.device, dtype=torch.long)
                        masks_t = masks_t[keep_idx_t]
                        if logits_t is not None and logits_t.shape[0] >= masks_t.shape[0]:
                            logits_t = logits_t[keep_idx_t]
                        scores_np = scores_np[keep_idx]

                # Ensure mask tensors are in original image size.
                if logits_t is not None and logits_t.shape[-2:] != (image_h, image_w):
                    logits_t = F.interpolate(
                        logits_t.unsqueeze(1),
                        size=(image_h, image_w),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                if logits_t is not None:
                    # sam3 processor may already return probability maps in [0, 1].
                    # Avoid applying sigmoid twice, which can shrink masks too much.
                    logits_min = float(logits_t.min().item())
                    logits_max = float(logits_t.max().item())
                    if 0.0 <= logits_min and logits_max <= 1.0:
                        probs_t = logits_t
                    else:
                        probs_t = logits_t.sigmoid()
                    masks_t = probs_t > 0.5
                elif masks_t.shape[-2:] != (image_h, image_w):
                    # Fallback for models that only provide binary masks.
                    masks_t = F.interpolate(
                        masks_t.unsqueeze(1),
                        size=(image_h, image_w),
                        mode="nearest",
                    )[:, 0] > 0.5

                masks_np = masks_t.cpu().numpy()

                if mask_strategy == 'merge':
                    selected_mask = masks_np.any(axis=0).squeeze()
                else:
                    # Default strategy: use the highest-score single instance to keep mask-object consistency.
                    if scores_np is not None and len(scores_np) > 0:
                        best_idx = int(np.argmax(scores_np))
                    else:
                        best_idx = 0
                    selected_mask = masks_np[best_idx].squeeze()

                if selected_mask.ndim == 3:  # Just in case it's [1, H, W]
                    selected_mask = selected_mask[0]
                selected_mask = selected_mask.astype(bool)
                
                # Show masks for debug
                image_np = np.array(image)
                if self.config.get("debug", False):
                    show_masks(
                        image_np,
                        masks.cpu().numpy(),
                        scores.detach().float().cpu().numpy(),
                        f"sam3_object_{object_idx:02d}_{object_name}",
                        borders=True,
                    )
                
                output_masks.append(selected_mask)
            else:
                print(f"Warning: No objects found for prompt '{object_name}'")
                output_masks.append(np.zeros((image.height, image.width), dtype=bool))
                
        return output_masks
