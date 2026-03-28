# Standard library
import os
import sys
import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Third-party
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.cluster import DBSCAN
from hezar.models import Model
from paddleocr import PaddleOCR

# Local path setup
sys.path.append(".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class OCR:
    def __init__(self, conf=0.6, tile_h=800, tile_w=800, tile_overlap=30, scale=1.0, max_workers=4, box_condense=(4,0)):
        self.craft = Model.load("hezarai/CRAFT", device="cuda", link_threshold=0.4)
        self.ocr = PaddleOCR(
            lang='en', 
            rec_char_type='en',
            use_angle_cls=False, 
            use_textline_orientation=False,
            rec_model_dir='./models/OCR/ch_PP-OCRv3_rec_small',
            rec_algorithm='CRNN',
            det_db_box_type="quad"
        )
        self.CONF_THRESH = conf
        self.scale = scale
        self.max_workers = max_workers
        self.box_condense = box_condense
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_overlap = tile_overlap

    def _hash_crop(self, crop: np.ndarray):
        return hashlib.md5(crop.tobytes()).hexdigest()


    def _process_tile(self, screenshot, left, top, right, bottom):
        """detect text boxes in a tile and map to global coords"""
        tile = screenshot.crop((left, top, right, bottom))
        boxes = self.craft.predict(tile)[0]["boxes"]
        tile_w, tile_h = right-left, bottom-top
        edge = 3
        out = []
        for x,y,w,h in boxes:
            l,t,r,b = x,y,x+w,y+h
            if (t<=edge or b>=tile_h-edge):
                continue
            out.append((int(l+left), int(t+top), int(w), int(h)))
        return out

    
    def _condense_boxes(self, boxes, scale, x_limit_px=3000):
        """merge nearby/overlapping boxes into larger ones"""
        if not boxes:
            return []

        x_tol = self.box_condense[0] * scale
        y_tol = self.box_condense[1] * scale

        boxes = [tuple(b) for b in boxes]
        changed = True

        while changed:
            changed = False
            new_boxes = []
            used = set()

            for i, box0 in enumerate(boxes):
                if i in used:
                    continue
                l0, t0, r0, b0 = box0[0], box0[1], box0[0]+box0[2], box0[1]+box0[3]
                h0 = box0[3]
                merged = box0

                for j, box1 in enumerate(boxes):
                    if j == i or j in used:
                        continue

                    l1, t1, r1, b1 = box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]
                    h1 = box1[3]

                    x_pass = not (r1 < l0 - x_tol or r0 < l1 - x_tol)
                    y_pass = abs(t1 - t0) <= y_tol and abs(h1 - h0) < min(h0, h1)

                    if x_pass and y_pass:
                        nl, nr = min(l0, l1), max(r0, r1)
                        nt, nb = min(t0, t1), max(b0, b1)
                        if nr - nl >= x_limit_px:
                            continue
                        merged = (nl, nt, nr - nl, nb - nt)
                        used.add(j)
                        changed = True
                        break
                new_boxes.append(merged)
                used.add(i)
            boxes = new_boxes
        return boxes
                    

    def split_text_vertically(self, crop, threshold=0.35, proj_noise=0.05,
                            edge_pad=3, visualize=False, target_px=300):
        """split a box into multiple lines using vertical projection"""

        h, w = crop.shape[:2]

        scale = max(target_px / min(h, w), 1.0)
        min_height = max(8, int(h * 0.06 * scale))

       # Resize and blur
        new_w, new_h = int(w * scale), int(h * scale)
        up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (15,15), sigmaX=35.0)
        up = cv2.addWeighted(up, 1.6, blur, -.5, 0)
        crop_up = cv2.bilateralFilter(up, d=1, sigmaColor=50, sigmaSpace=50)

        # Compute gradient mag
        gray = cv2.cvtColor(crop_up, cv2.COLOR_RGB2GRAY)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag /= mag.max() + 1e-6
        edges = mag > threshold

        proj = edges.sum(axis=1)
        proj_clean = np.where(proj >= proj.max() * proj_noise, proj, 0)

        # Detect continuous segments as boxes
        diff = np.diff(np.pad((proj_clean > 0).astype(int), (1, 1)))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

        boxes = []
        pad = int(round(edge_pad * scale))

        for y0, y1 in zip(starts, ends):
            if y1 - y0 < min_height:
                continue
            y0s = max(0, y0 - pad)
            y1s = min(new_h, y1 + pad)
            if y1s <= y0s:
                continue
            yo = y0s
            yb = y1s
            ho = yb - yo
            if ho <= 1 or yo < 0 or yo + ho > h:
                continue
            boxes.append((0, yo, w, ho))

        if visualize and boxes:
            plt.imshow(crop)
            for x, y, bw, bh in boxes:
                plt.gca().add_patch(plt.Rectangle((x, y), bw, bh, fill=False))
            plt.show()

        return boxes
    

    def group_lines_dbscan(self, items):
        """cluster boxes into text lines using DBSCAN"""
        # items=[(box, text, crop),...]
        y_mids = np.array([b[1] + b[3] / 2 for b, _, _ in items])
        heights = np.array([b[3] for b, _, _ in items])

        # Normalize y by height → scale-invariant grouping
        y_norm = y_mids / (heights.mean() + 1e-6)

        clustering = DBSCAN(eps=0.4, min_samples=1).fit(y_norm.reshape(-1, 1))
        clusters = {}
        for label, item in zip(clustering.labels_, items):
            clusters.setdefault(label, []).append(item)

        sorted_clusters = sorted(
            clusters.values(),
            key=lambda cluster: np.mean([b[1] + b[3] / 2 for b, _, _ in cluster])
        )
        return sorted_clusters


    def apply_vertical_splitting(self, boxes, screenshot):
        """ apply vertical splitting to all detected boxes """
        final_boxes = []
        for x, y, w, h in boxes:
            if w <= 1 or h <= 1: continue
            r, b = int(x+w), int(y+h)
            if r <= int(x) or b <= int(y): continue

            crop = np.array(screenshot.crop((int(x), int(y), r, b)))
            sub_boxes = self.split_text_vertically(crop, visualize=False)
            if not sub_boxes:
                final_boxes.append((x, y, w, h))
                continue
            for x0, y0, w0, h0 in sub_boxes:
                final_boxes.append((x + x0, y + y0, w0, h0))
        return final_boxes


    def remove_duplicate_boxes(self, boxes, texts, crops, iou_thresh=0.5, size_tol=0.3):
        """remove overlapping duplicate boxes with same text"""
        def iou(a, b): # Intersection over Union
            xa, ya = max(a[0], b[0]), max(a[1], b[1])
            xb, yb = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
            inter = max(0, xb-xa) * max(0, yb-ya)
            return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-6)

        def similar(a, b):
            return (abs(a[2]-b[2])/max(a[2], b[2]) < size_tol and
                    abs(a[3]-b[3])/max(a[3], b[3]) < size_tol)

        keep, used = [], set()
        for i, (b1, t1) in enumerate(zip(boxes, texts)):
            if i in used:
                continue
            for j in range(i+1, len(boxes)):
                if j in used:
                    continue
                if t1 == texts[j] and iou(b1, boxes[j]) > iou_thresh and similar(b1, boxes[j]):
                    used.add(j)
            keep.append(i)
        return [boxes[i] for i in keep], [texts[i] for i in keep], [crops[i] for i in keep]


    def visualize_split(self, screenshot, boxes, final_boxes):
        """visualize boxes before and after splitting"""
        img = np.array(screenshot)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img)

        for x, y, w, h in boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title("Before (original boxes)")
        axes[0].axis('off')

        axes[1].imshow(img)
        for x, y, w, h in final_boxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            axes[1].add_patch(rect)
        axes[1].set_title("After (split_text_vertically)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


    def show_boxes(self, screenshot, boxes):
        """draw and display bounding boxes on image"""
        img_np = np.array(screenshot)
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img_np)

        for i, (x, y, w, h) in enumerate(boxes):
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, max(y-5, 0), str(i), color='black', fontsize=20)
        plt.axis('off')
        plt.show()



    def __call__(self, screenshot: Image):
        """full OCR pipeline (detect → refine → recognize → group)"""
        W, H = screenshot.size 
        scale = self.scale
        screenshot = screenshot.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
        W, H = screenshot.size 

        all_boxes = [] 
        tiles = [
            (left, top, min(left + self.tile_w, W), min(top + self.tile_h, H))
            for top in range(0, H, max(1, self.tile_h - self.tile_overlap))
            for left in range(0, W, max(1, self.tile_w - self.tile_overlap))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe: 
            results = exe.map(lambda args: self._process_tile(screenshot, *args), tiles)
        
        for res in results: 
            all_boxes.extend(res) # [x,y,w,h]

        boxes = self._condense_boxes(all_boxes, scale)
        #self.visualize_split(screenshot, all_boxes, boxes)

        boxes = self.apply_vertical_splitting(boxes, screenshot)
        #self.visualize_split(screenshot, boxes, boxes)

        crops = [np.array(screenshot.crop((int(x), int(y), int(x+w), int(y+h)))) 
                for x, y, w, h in boxes]

        # Processing crops 
        def process_crop(box, crop):
            result = self.ocr.ocr(crop, det=False, cls=False)
            if not result or not result[0] or not result[0][0]:
                return "", crop
            text, conf = result[0][0][0].strip(), result[0][0][1]
            if conf < self.CONF_THRESH:
                text = ""
            return text, crop

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            processed_crops = list(ex.map(lambda args: process_crop(*args), zip(boxes, crops)))

        # post-process filtering
        filtered = [(box, text, crop)
                    for box, (text, crop) in zip(boxes, processed_crops)
                    if text and text.strip() != ""]
        if not filtered: 
            return []
     
        boxes, texts, crops_out = zip(*filtered)

        boxes = list(boxes)
        texts = list(texts)
        crops_out = list(crops_out)

        #self.show_boxes(screenshot, boxes)
        boxes, texts, crops_out = self.remove_duplicate_boxes(boxes, texts, crops_out)

        # Restore scaling
        boxes = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x, y, w, h in boxes]

        # Sort and group lines
        items = sorted(zip(boxes, texts, crops_out), key=lambda x: (x[0][1]+x[0][3]//2, x[0][0]))
        Y_lines = self.group_lines_dbscan(items)

        final_lines = []
        for line in Y_lines:
            line = sorted(line, key=lambda x: x[0][0])
            prev_x_end = None
            avg_w = np.mean([b[2] for b, _, _ in line])
            new_line = []
            for i, (b, t, c) in enumerate(line):
                x_start, x_end = b[0], b[0] + b[2]
                t = (" " if prev_x_end is not None and x_start - prev_x_end > avg_w * 0.5 else "") + t.strip()
                if i == len(line) - 1:
                    t += "\n"
                new_line.append((b, t, c))
                prev_x_end = x_end
            final_lines.append(new_line)

        #printlines = ["".join(t for _, t, _ in line) for line in final_lines]
        #print("final lines: ", printlines)

        return final_lines