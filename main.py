#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:49:57 2019

@author: xingyu
"""
import sys
sys.path.append('./LPRNet')
sys.path.append('./MTCNN')
from LPRNet.LPRNet_Test import *
import numpy as np
import argparse
import torch
import time
import imghdr
import cv2, os
from pathlib import Path
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'JPEG', 'JPG', 'PNG'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if imghdr.what(file_path) in img_end or imghdr.what(file_path) == None:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    return imgs_lists


@torch.no_grad()
def create_yolov5_net(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    webcam = False
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()


    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    im0s = cv2.imread(source)
    im = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=augment, visualize=False)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(pred, '--------------------')
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        if webcam:  # batch_size >= 1
            im0 = im0s[i].copy()
        else:
            im0 = im0s.copy()

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            return det


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='/data/zjw_plus/code/yolov5/runs/train/weight/best.pt', help='model path(s)')
    parser.add_argument('--source', '--image', type=str, default='/data/zjw_plus/code/yolov5/data/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default='/data/zjw_plus/code/yolov5/data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.60, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument("-draw_img_save", help='image path', default='/data/zjw_plus/code/yolov5/runs/results/', type=str)
    parser.add_argument("--scale", dest='scale', help="scale the iamge", default=1, type=int)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt



if __name__ == '__main__':

    args = parse_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('/data/zjw_plus/code/LPD/LPRNet/weights/firday_enhance_green_lprnet_model.pth', map_location=lambda storage, loc: storage))
    lprnet.eval()
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('/data/zjw_plus/code/LPD/LPRNet/weights/firday_enhance_green_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()
    
    print("Successful to build LPR network!")
    image_file_list = get_image_file_list(args.source)
    for image_file in image_file_list:
        since = time.time()
        image = cv2.imread(image_file)
        bboxes = create_yolov5_net(weights=args.weights,  # model.pt path(s)
        source=image_file,  # file/dir/URL/glob, 0 for webcam
        data=args.data,  # dataset.yaml path
        imgsz=args.imgsz,  # inference size (height, width)
        conf_thres=args.conf_thres,  # confidence threshold
        iou_thres=args.iou_thres,  # NMS IOU threshold
        max_det=args.max_det,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,)  # use OpenCV DNN for ONNX inference)
        if bboxes == None:
            cv2.imwrite(os.path.join(args.draw_img_save, os.path.basename(image_file)), image)
            continue
        bboxes = bboxes.cpu().numpy()
        bboxes = bboxes[:, 0:-1]
        image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)


        for i in range(bboxes.shape[0]):

            bbox = bboxes[i, :4]
            x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
            w = int(x2 - x1 + 1.0)
            h = int(y2 - y1 + 1.0)
            img_box = np.zeros((h, w, 3))
            img_box = image[y1:y2+1, x1:x2+1, :]
            im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
            im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
            data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])
            transfer = STN(data)
            preds = lprnet(transfer)
            preds = preds.cpu().detach().numpy()  # (1, 68, 18)
            labels, pred_labels = decode(preds, CHARS)
            print('--------', labels, pred_labels)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            image = cv2ImgAddText(image, labels[0], (x1, y1-12), textColor=(255, 255, 0), textSize=15)

        print("model inference in {:2.3f} seconds".format(time.time() - since))
        image = cv2.resize(image, (0, 0), fx = 1/args.scale, fy = 1/args.scale, interpolation=cv2.INTER_CUBIC)
        print(os.path.join(args.draw_img_save, os.path.basename(image_file)))
        cv2.imwrite(os.path.join(args.draw_img_save, os.path.basename(image_file)), image)