import os
import cv2
import glob
import torch
import numpy as np

folder = '/home/ubuntu/datasets/yoga/1-1/cam3_high/'
path = folder + 'color/*.jpg'

import sys
# Yolov3
sys.path.append('/home/ubuntu/object_detection/common/yolov3')
from models.experimental import attempt_load
from utils.general import check_img_size, set_logging
from utils.torch_utils import select_device
sys.path.append('/home/ubuntu/object_detection/person/yolov3_person')
from yolov3_person import person_detection

def yolov3_init(img_size=640, device=''):
    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    # Load model
    yolov3_model = attempt_load(weights='./yolov3.pt', map_location=device)
    imgsz = check_img_size(img_size, s=yolov3_model.stride.max())
    if half:
        yolov3_model.half()

    # Warm up
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = yolov3_model(img.half() if half else img) if device.type != 'cpu' else None

    return yolov3_model, device, half, imgsz


def main():
    # Yolov3 initialization
    yolov3_model, device, half, imgsz = yolov3_init()

    # Inference
    bbox_list = []
    idx = 0
    for p in sorted(glob.glob(path)):
        # print(os.path.basename(p).split('_')[1], idx)
        img = cv2.imread(p)
        bbox = person_detection(yolov3_model, img, imgsz, device, half)
        # for b in bbox:
        #     cv2.rectangle(img, (b[0],b[1]), (b[2], b[3]), (0,0,255), 2)
        # cv2.imshow('test',img)
        # if cv2.waitKey(100) == 27:
        #     cv2.destroyAllWindows()
        #     break
        bbox_list.append(bbox)
        idx += 1
    np.save(folder+'bbox.npy', np.array(bbox_list))

if __name__ == '__main__':
    with torch.no_grad():
        main()
