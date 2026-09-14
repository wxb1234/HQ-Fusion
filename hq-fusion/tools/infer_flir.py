import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)

class_ = ['car', 'person', 'bicycle']
colors_ = [
    (0, 255, 0), 
    (0, 0, 255), 
    (255, 0, 0), 
]

# draw_yolo_style
def draw(images, labels, boxes, scores, conf_thres=0.6, filename=''):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        lab = labels[0]
        box = boxes[0]
        scrs = scores[0]

        for j in range(len(box)):
            if scrs[j] < conf_thres:
                continue

            b = box[j]
            cls = int(lab[j])
            conf = float(scrs[j])

            x1, y1, x2, y2 = map(int, b)
            color = colors_[cls]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{class_[cls]} {conf:.2f}"
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            draw.rectangle(
                [x1, y1 - text_h - 3, x1 + text_w + 2, y1],
                fill=color
            )

            draw.text(
                (x1 + 1, y1 - text_h - 2),
                label,
                fill=(255, 255, 255),
                font=font
            )

        im.save(f"/home/cx/Projects/wen/code/point2/RT-DETR-main/z_flir_v/{filename}_{i}.jpg")


import thop
from copy import deepcopy
def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    try:
        # model = de_parallel(model)
        # p = next(model.parameters())
        stride = 640
        im = torch.empty((1, 3, stride, stride), device='cuda')  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=[None,im,im], verbose=False)[0] / 1E9 * 2 if thop else 0  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        return flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception as e:
        print(e)
        return 0

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, images1, orig_target_sizes):
            outputs = self.model(images, images1)
            outputs = self.postprocessor(outputs, orig_target_sizes) # 输出框xywh->x1y1x2y2
            return outputs
    
    model = Model().to(args.device)

    # 遍历文件夹
    for name in os.listdir(args.im_folder):
        full = os.path.join(args.im_folder, name)
        if os.path.isfile(full):
            im_pil = Image.open(full).convert('RGB')

            ir_file = full.replace('images', 'imagesIR')
            ir_pil = Image.open(ir_file).convert('RGB')
            
            w, h = im_pil.size
            orig_size = torch.tensor([w, h])[None].to(args.device)
            
            transforms = T.Compose([
                T.Resize((640, 640)),  
                T.ToTensor(),
            ])
            im_data = transforms(im_pil)[None].to(args.device)
            im_ir_data = transforms(ir_pil)[None].to(args.device)
            # im_data = torch.cat((im_ir_data, im_data), dim=1)
            output = model(im_data, im_ir_data, orig_size)
            labels, boxes, scores = output

            draw([im_pil, ir_pil], labels, boxes, scores, 0.6, name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco-flir.yml')
    parser.add_argument('-r', '--resume', type=str, default='output/r50-flir/checkpoint.pth')
    
    parser.add_argument('-f', '--im-file', type=str, default='/home/cx/Projects/wen/dataSets/flir_align/images/test/')
    parser.add_argument('-folder', '--im-folder', type=str, default='/home/cx/Projects/wen/dataSets/flir_align/images/test/')
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)
