"""FlashTL-1K contextual feature extraction.

Extracts spatial, temporal, depth, scene, and caption-based contextual
attributes from FlashTL-1K videos. Results and summary statistics are saved
to a single Excel workbook.

Installation and command-line usage are documented in the repository README.
"""
import os, sys, gc, json, time, logging, argparse, warnings, traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
matplotlib.use('Agg')
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
warnings.filterwarnings('ignore')
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline, CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration
    from PIL import Image
    TRANSFORMERS_OK = True
except ImportError:
    TRANSFORMERS_OK = False
    print('[WARNING] transformers not installed. pip install transformers accelerate')
try:
    import supervision as sv
    SUPERVISION_OK = True
except ImportError:
    SUPERVISION_OK = False
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_OK = True
except ImportError:
    SAM2_OK = False
    print('[WARNING] SAM2 not installed. pip install git+https://github.com/facebookresearch/segment-anything-2.git')
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
CLASS_NAMES = ['Blinking_Red', 'Blinking_Yellow']
CLIP_SCENE_PROMPTS = ['daytime road scene', 'nighttime road scene', 'rainy road scene', 'inside vehicle dashcam view', 'outside traffic intersection', 'clear weather road', 'traffic light blinking red', 'traffic light blinking yellow']
GROUNDING_DINO_QUERIES = 'traffic light . red light . yellow light . blinking light .'
DEPTH_MODEL = 'depth-anything/Depth-Anything-V2-Small-hf'
CLIP_MODEL = 'openai/clip-vit-base-patch32'
BLIP2_MODEL = 'Salesforce/blip2-opt-2.7b'
GDINO_MODEL = 'IDEA-Research/grounding-dino-tiny'

def get_args():
    p = argparse.ArgumentParser(description='FlashTL-1K Contextual Feature Extraction Pipeline')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./context_outputs')
    p.add_argument('--csv_path', type=str, default=None, help='Use existing dataset CSV (from train_recognition.py)')
    p.add_argument('--sample', type=int, default=None, help='Process only N videos per class (for testing)')
    p.add_argument('--num_frames', type=int, default=8, help='Frames to sample per clip for context extraction')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--skip_gdino', action='store_true', help='Skip Grounding DINO')
    p.add_argument('--skip_sam', action='store_true', help='Skip SAM2')
    p.add_argument('--skip_depth', action='store_true', help='Skip Depth Anything V2')
    p.add_argument('--skip_fft', action='store_true', help='Skip FFT analysis')
    p.add_argument('--skip_clip', action='store_true', help='Skip CLIP')
    p.add_argument('--skip_blip2', action='store_true', help='Skip BLIP-2')
    args = p.parse_args()
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def infer_label(path: str) -> Tuple[int, str]:
    p = path.lower().replace('\\', '/')
    if 'red' in p:
        return (0, 'Blinking_Red')
    if 'yellow' in p:
        return (1, 'Blinking_Yellow')
    raise ValueError(f'Cannot infer label: {path}')

def parse_conditions(path: str) -> Dict:
    """
    Parses scene/time-of-day from folder names if present.
    With the new flat structure (Red_Blinking/, Yellow_Blinking/ only),
    these will return 'unknown' — that's expected, not an error.
    CLIP-based scene tagging (clip_is_day, clip_is_night, clip_is_rain)
    now provides this info instead, inferred directly from the video content.
    """
    p = path.lower().replace('\\', '/')
    return {'scene': 'inside' if 'inside' in p else 'outside' if 'outside' in p else 'rain' if 'rain' in p else 'unknown', 'time_of_day': 'day' if 'day' in p else 'night' if 'night' in p else 'unknown'}

def scan_dataset(data_root: str, csv_path: Optional[str], sample: Optional[int]) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        print(f'Loaded CSV: {csv_path}  ({len(df)} clips)')
    else:
        records = []
        for root, _, files in os.walk(data_root):
            for f in sorted(files):
                if Path(f).suffix.lower() in VIDEO_EXTS:
                    fp = os.path.join(root, f)
                    try:
                        li, ln = infer_label(fp)
                    except ValueError:
                        continue
                    cond = parse_conditions(fp)
                    records.append({'path': fp, 'label': li, 'label_name': ln, **cond})
        df = pd.DataFrame(records)
        print(f'Scanned: {len(df)} clips')
    if sample:
        df = df.groupby('label_name').apply(lambda g: g.sample(min(len(g), sample), random_state=42)).reset_index(drop=True)
        print(f'Sampled: {len(df)} clips ({sample} per class)')
    return df

def read_video_frames(path: str, n: int=8, full: bool=False) -> Tuple[List[np.ndarray], float]:
    """
    Returns (frames_list, fps).
    frames_list: list of RGB uint8 ndarrays.
    If full=True, returns all frames (for FFT).
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    if full:
        idxs = list(range(total))
    else:
        idxs = np.linspace(0, total - 1, n, dtype=int).tolist()
    frames, last = ([], None)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last = frame
        else:
            frame = last if last is not None else np.zeros((360, 640, 3), dtype=np.uint8)
        frames.append(frame)
    cap.release()
    return (frames, fps)

class GroundingDINOExtractor:

    def __init__(self, device: str):
        print('  Loading Grounding DINO...')
        self.processor = AutoProcessor.from_pretrained(GDINO_MODEL)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
        self.model.eval()
        self.device = device
        print('  ✓ Grounding DINO ready')

    @torch.no_grad()
    def extract(self, frames: List[np.ndarray], video_path: str) -> Dict:
        """Run on middle frame + first/last for robustness."""
        result = {'gdino_detected': False, 'gdino_num_detections': 0, 'gdino_max_confidence': 0.0, 'gdino_mean_confidence': 0.0, 'gdino_bbox_x1': None, 'gdino_bbox_y1': None, 'gdino_bbox_x2': None, 'gdino_bbox_y2': None, 'gdino_bbox_cx': None, 'gdino_bbox_cy': None, 'gdino_bbox_area_ratio': None, 'gdino_position': None, 'gdino_all_boxes': []}
        frame = frames[len(frames) // 2]
        pil_img = Image.fromarray(frame)
        H, W = frame.shape[:2]
        inputs = self.processor(images=pil_img, text=GROUNDING_DINO_QUERIES, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        try:
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[(H, W)])[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.35, text_threshold=0.25, target_sizes=[(H, W)])[0]
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        MIN_AREA_RATIO = 0.0005
        if len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (W * H)
            keep = areas >= MIN_AREA_RATIO
            boxes, scores = (boxes[keep], scores[keep])
        if len(boxes) == 0:
            return result
        best = int(np.argmax(scores))
        bx1, by1, bx2, by2 = boxes[best]
        cx = (bx1 + bx2) / 2 / W
        cy = (by1 + by2) / 2 / H
        area_ratio = (bx2 - bx1) * (by2 - by1) / (W * H)
        pos_h = 'top' if cy < 0.33 else 'middle' if cy < 0.67 else 'bottom'
        pos_w = 'left' if cx < 0.33 else 'centre' if cx < 0.67 else 'right'
        position = f'{pos_h}-{pos_w}'
        result.update({'gdino_detected': True, 'gdino_num_detections': len(boxes), 'gdino_max_confidence': float(scores.max()), 'gdino_mean_confidence': float(scores.mean()), 'gdino_bbox_x1': float(bx1), 'gdino_bbox_y1': float(by1), 'gdino_bbox_x2': float(bx2), 'gdino_bbox_y2': float(by2), 'gdino_bbox_cx': float(cx), 'gdino_bbox_cy': float(cy), 'gdino_bbox_area_ratio': float(area_ratio), 'gdino_position': position, 'gdino_all_boxes': boxes.tolist()})
        return result

class SAM2Extractor:

    def __init__(self, device: str):
        if not SAM2_OK:
            raise RuntimeError('SAM2 not installed')
        print('  Loading SAM2...')
        self.predictor = build_sam2_video_predictor('sam2_hiera_small.yaml', 'sam2_hiera_small.pt', device=device)
        self.device = device
        print('  ✓ SAM2 ready')

    def extract(self, frames: List[np.ndarray], bbox: Optional[List[float]]) -> Dict:
        result = {'sam2_mask_area_ratio': None, 'sam2_mask_iou_across_frames': None}
        if bbox is None:
            return result
        try:
            mid = len(frames) // 2
            with torch.inference_mode():
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                pred = SAM2ImagePredictor(self.predictor.model)
                pred.set_image(frames[mid])
                masks, scores, _ = pred.predict(box=np.array(bbox), multimask_output=False)
                mask = masks[0]
                area_ratio = float(mask.sum()) / (mask.shape[0] * mask.shape[1])
                result['sam2_mask_area_ratio'] = area_ratio
        except Exception as e:
            result['sam2_error'] = str(e)
        return result

class DepthExtractor:

    def __init__(self, device: str):
        print('  Loading Depth Anything V2...')
        self.pipe = pipeline(task='depth-estimation', model=DEPTH_MODEL, device=0 if device == 'cuda' else -1)
        self.device = device
        print('  ✓ Depth Anything V2 ready')

    @torch.no_grad()
    def extract(self, frames: List[np.ndarray], bbox: Optional[List[float]]) -> Dict:
        result = {'depth_mean': None, 'depth_at_bbox': None, 'depth_at_bbox_percentile': None, 'depth_min': None, 'depth_max': None, 'depth_std': None}
        frame = frames[len(frames) // 2]
        pil_img = Image.fromarray(frame)
        try:
            out = self.pipe(pil_img)
            depth_map = np.array(out['depth'])
            d_min, d_max = (depth_map.min(), depth_map.max())
            if d_max > d_min:
                depth_norm = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_norm = depth_map
            result.update({'depth_mean': float(depth_norm.mean()), 'depth_min': float(depth_norm.min()), 'depth_max': float(depth_norm.max()), 'depth_std': float(depth_norm.std())})
            if bbox is not None:
                H, W = depth_map.shape
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = (max(0, x1), max(0, y1))
                x2, y2 = (min(W, x2), min(H, y2))
                if x2 > x1 and y2 > y1:
                    roi = depth_norm[y1:y2, x1:x2]
                    roi_mean = float(roi.mean())
                    result['depth_at_bbox'] = roi_mean
                    result['depth_at_bbox_percentile'] = float((depth_norm < roi_mean).mean() * 100)
            self._last_depth = depth_norm
        except Exception as e:
            result['depth_error'] = str(e)
            self._last_depth = None
        return result

class FFTExtractor:
    """
    Extracts the dominant flash frequency of the traffic light.
    Method:
      1. Read ALL frames of the video
      2. Crop the bbox region (or use centre crop if no bbox)
      3. Compute mean brightness per frame → intensity signal I(t)
      4. FFT → find dominant frequency peak
    """

    def extract(self, video_path: str, fps: float, bbox: Optional[List[float]]) -> Dict:
        result = {'fft_dominant_freq_hz': None, 'fft_dominant_period_sec': None, 'fft_peak_power': None, 'fft_signal_mean': None, 'fft_signal_std': None, 'fft_flicker_ratio': None}
        try:
            frames, fps_actual = read_video_frames(video_path, n=8, full=True)
            fps_use = fps_actual if fps_actual > 1 else fps
            intensities = []
            for frame in frames:
                H, W = frame.shape[:2]
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1, y1 = (max(0, x1), max(0, y1))
                    x2, y2 = (min(W, x2), min(H, y2))
                    roi = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
                else:
                    qH, qW = (H // 4, W // 4)
                    roi = frame[qH:3 * qH, qW:3 * qW]
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                intensities.append(float(gray.mean()))
            if len(intensities) < 4:
                return result
            sig = np.array(intensities, dtype=np.float32)
            sig -= sig.mean()
            N = len(sig)
            fft = np.fft.rfft(sig)
            freqs = np.fft.rfftfreq(N, d=1.0 / fps_use)
            power = np.abs(fft) ** 2
            if len(power) > 1:
                peak_idx = int(np.argmax(power[1:]) + 1)
                dom_freq = float(freqs[peak_idx])
                dom_period = 1.0 / dom_freq if dom_freq > 0 else None
                peak_power = float(power[peak_idx])
            else:
                dom_freq = dom_period = peak_power = None
            sig_orig = np.array(intensities)
            flicker = float(sig_orig.std()) / float(sig_orig.mean()) if sig_orig.mean() > 0 else 0.0
            result.update({'fft_dominant_freq_hz': dom_freq, 'fft_dominant_period_sec': dom_period, 'fft_peak_power': peak_power, 'fft_signal_mean': float(sig_orig.mean()), 'fft_signal_std': float(sig_orig.std()), 'fft_flicker_ratio': flicker, '_fft_signal': intensities, '_fft_freqs': freqs.tolist(), '_fft_power': power.tolist()})
        except Exception as e:
            result['fft_error'] = str(e)
        return result

class CLIPExtractor:

    def __init__(self, device: str):
        print('  Loading CLIP...')
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
        self.model.eval()
        self.device = device
        print('  ✓ CLIP ready')

    @torch.no_grad()
    def extract(self, frames: List[np.ndarray]) -> Dict:
        result = {f"clip_{p.replace(' ', '_')}": None for p in CLIP_SCENE_PROMPTS}
        result.update({'clip_top_scene': None, 'clip_top_score': None, 'clip_is_day': None, 'clip_is_night': None, 'clip_is_rain': None, 'clip_is_inside': None})
        try:
            pil_frames = [Image.fromarray(f) for f in frames]
            all_scores = []
            for pil_img in pil_frames:
                inputs = self.processor(text=CLIP_SCENE_PROMPTS, images=pil_img, return_tensors='pt', padding=True).to(self.device)
                out = self.model(**inputs)
                probs = out.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                all_scores.append(probs)
            mean_scores = np.mean(all_scores, axis=0)
            top_idx = int(np.argmax(mean_scores))
            for i, prompt in enumerate(CLIP_SCENE_PROMPTS):
                result[f"clip_{prompt.replace(' ', '_')}"] = float(mean_scores[i])
            result['clip_top_scene'] = CLIP_SCENE_PROMPTS[top_idx]
            result['clip_top_score'] = float(mean_scores[top_idx])
            day_idx = CLIP_SCENE_PROMPTS.index('daytime road scene')
            night_idx = CLIP_SCENE_PROMPTS.index('nighttime road scene')
            rain_idx = CLIP_SCENE_PROMPTS.index('rainy road scene')
            inside_idx = CLIP_SCENE_PROMPTS.index('inside vehicle dashcam view')
            result['clip_is_day'] = bool(mean_scores[day_idx] > mean_scores[night_idx])
            result['clip_is_night'] = bool(mean_scores[night_idx] > mean_scores[day_idx])
            result['clip_is_rain'] = bool(mean_scores[rain_idx] > 0.15)
            result['clip_is_inside'] = bool(mean_scores[inside_idx] > 0.15)
        except Exception as e:
            result['clip_error'] = str(e)
        return result

class BLIP2Extractor:

    def __init__(self, device: str):
        print('  Loading BLIP-2 (this may take a minute)...')
        self.processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
        self.model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_MODEL, torch_dtype=torch.float16 if device == 'cuda' else torch.float32).to(device)
        self.model.eval()
        self.device = device
        print('  ✓ BLIP-2 ready')
    QUESTIONS = ['What type of traffic light is visible?', 'Is it daytime or nighttime?', 'What is the weather condition?', 'Describe the scene briefly.']

    @torch.no_grad()
    def extract(self, frames: List[np.ndarray]) -> Dict:
        result = {'blip2_caption': None, 'blip2_traffic_light_qa': None, 'blip2_daytime_qa': None, 'blip2_weather_qa': None, 'blip2_scene_qa': None}
        frame = frames[len(frames) // 2]
        pil_img = Image.fromarray(frame)
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        try:
            inputs = self.processor(pil_img, return_tensors='pt').to(self.device, dtype)
            gen = self.model.generate(**inputs, max_new_tokens=60)
            caption = self.processor.decode(gen[0], skip_special_tokens=True)
            result['blip2_caption'] = caption.strip()
            qa_keys = ['blip2_traffic_light_qa', 'blip2_daytime_qa', 'blip2_weather_qa', 'blip2_scene_qa']
            for key, question in zip(qa_keys, self.QUESTIONS):
                prompt = f'Question: {question} Answer:'
                inputs = self.processor(pil_img, text=prompt, return_tensors='pt').to(self.device, dtype)
                gen = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
                answer = self.processor.decode(gen[0], skip_special_tokens=True).strip()
                if answer.lower().startswith(question.lower()):
                    answer = answer[len(question):].strip(' :.')
                result[key] = answer
        except Exception as e:
            result['blip2_error'] = str(e)
        return result

def compute_summary_statistics(df_results: pd.DataFrame) -> Dict:
    """Compute dataset-level statistics for the paper."""
    stats = {}
    for label in CLASS_NAMES:
        sub = df_results[df_results['label_name'] == label]
        stats[label] = {}
        if 'gdino_detected' in sub.columns:
            stats[label]['detection_rate'] = float(sub['gdino_detected'].mean())
            stats[label]['mean_confidence'] = float(sub['gdino_max_confidence'].mean())
            stats[label]['mean_bbox_area'] = float(sub['gdino_bbox_area_ratio'].mean())
            if 'gdino_position' in sub.columns:
                pos_dist = sub['gdino_position'].value_counts(normalize=True).to_dict()
                stats[label]['position_distribution'] = pos_dist
        if 'depth_at_bbox' in sub.columns:
            d = sub['depth_at_bbox'].dropna()
            if len(d):
                stats[label]['mean_depth_at_light'] = float(d.mean())
                stats[label]['std_depth_at_light'] = float(d.std())
        if 'fft_dominant_freq_hz' in sub.columns:
            f = sub['fft_dominant_freq_hz'].dropna()
            if len(f):
                stats[label]['mean_flash_freq_hz'] = float(f.mean())
                stats[label]['std_flash_freq_hz'] = float(f.std())
                stats[label]['median_flash_freq_hz'] = float(f.median())
            fr = sub['fft_flicker_ratio'].dropna()
            if len(fr):
                stats[label]['mean_flicker_ratio'] = float(fr.mean())
        if 'clip_is_day' in sub.columns:
            stats[label]['pct_day'] = float(sub['clip_is_day'].mean())
            stats[label]['pct_night'] = float(sub['clip_is_night'].mean())
            stats[label]['pct_rain'] = float(sub['clip_is_rain'].mean())
            stats[label]['pct_inside'] = float(sub['clip_is_inside'].mean())
    return stats

def setup_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger('ctx_extract')
    log.setLevel(logging.INFO)
    if log.handlers:
        log.handlers.clear()
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(fh)
    log.addHandler(sh)
    return log

def main():
    args = get_args()
    out_dir = Path(args.output_dir)
    log = setup_logger(out_dir / 'extraction.log')
    log.info('=' * 65)
    log.info('  FlashTL-1K Contextual Feature Extraction Pipeline')
    log.info('=' * 65)
    log.info(f'  Device     : {args.device}')
    log.info(f'  Frames/clip: {args.num_frames}')
    log.info(f"  Sample     : {args.sample or 'ALL'}")
    log.info(f'  Output     : {out_dir}')
    df = scan_dataset(args.data_root, args.csv_path, args.sample)
    if len(df) == 0:
        raise RuntimeError(f'No videos found under {args.data_root}')
    gdino = GroundingDINOExtractor(args.device) if not args.skip_gdino and TRANSFORMERS_OK else None
    sam2 = None
    if not args.skip_sam and SAM2_OK:
        try:
            sam2 = SAM2Extractor(args.device)
        except Exception as e:
            log.info(f'  SAM2 load failed: {e}')
    depth = DepthExtractor(args.device) if not args.skip_depth and TRANSFORMERS_OK else None
    fft = FFTExtractor() if not args.skip_fft else None
    clip_ = CLIPExtractor(args.device) if not args.skip_clip and TRANSFORMERS_OK else None
    blip2 = BLIP2Extractor(args.device) if not args.skip_blip2 and TRANSFORMERS_OK else None
    log.info(f'\n  Active modules:')
    log.info(f"    Grounding DINO : {('✓' if gdino else '✗')}")
    log.info(f"    SAM2           : {('✓' if sam2 else '✗')}")
    log.info(f"    Depth Anything : {('✓' if depth else '✗')}")
    log.info(f"    FFT Analysis   : {('✓' if fft else '✗')}")
    log.info(f"    CLIP           : {('✓' if clip_ else '✗')}")
    log.info(f"    BLIP-2         : {('✓' if blip2 else '✗')}")
    all_results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Extracting context'):
        video_path = row['path']
        label_name = row['label_name']
        video_name = Path(video_path).stem
        scene = row.get('scene', 'unknown')
        tod = row.get('time_of_day', 'unknown')
        record = {'video_path': video_path, 'video_name': video_name, 'label': row['label'], 'label_name': label_name, 'scene': scene, 'time_of_day': tod}
        try:
            frames, fps = read_video_frames(video_path, n=args.num_frames)
            bbox = None
            if gdino:
                r = gdino.extract(frames, video_path)
                record.update(r)
                if r.get('gdino_detected'):
                    bbox = [r['gdino_bbox_x1'], r['gdino_bbox_y1'], r['gdino_bbox_x2'], r['gdino_bbox_y2']]
            if sam2:
                r = sam2.extract(frames, bbox)
                record.update(r)
            if depth:
                r = depth.extract(frames, bbox)
                record.update(r)
            if fft:
                r = fft.extract(video_path, fps, bbox)
                record.update(r)
            if clip_:
                r = clip_.extract(frames)
                record.update(r)
            if blip2:
                r = blip2.extract(frames)
                record.update(r)
            log.info(f"  [{idx + 1}/{len(df)}] {video_name}  det={record.get('gdino_detected', '?')}  freq={record.get('fft_dominant_freq_hz', '?')}Hz  scene={record.get('clip_top_scene', '?')}")
        except Exception as e:
            log.info(f'  [ERROR] {video_name}: {e}')
            record['error'] = str(e)
        all_results.append(record)
    df_res = pd.DataFrame(all_results)

    result_columns = [
        c for c in df_res.columns
        if not c.startswith("_fft_") and not c.endswith("_all_boxes")
    ]
    results = df_res[result_columns].copy()

    stats = compute_summary_statistics(df_res)
    summary_rows = []
    for label, values in stats.items():
        for metric, value in values.items():
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    summary_rows.append({
                        "Class": label,
                        "Metric": f"{metric}.{submetric}",
                        "Value": subvalue,
                    })
            else:
                summary_rows.append({
                    "Class": label,
                    "Metric": metric,
                    "Value": value,
                })
    summary = pd.DataFrame(summary_rows)

    excel_path = out_dir / "context_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Context Results", index=False)
        summary.to_excel(writer, sheet_name="Summary Statistics", index=False)

    log.info("=" * 65)
    log.info("EXTRACTION COMPLETE")
    log.info(f"Excel results: {excel_path}")
    log.info(f"Log: {out_dir / 'extraction.log'}")
    log.info("=" * 65)

if __name__ == '__main__':
    main()
