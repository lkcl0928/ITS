import cv2
from PIL import Image, ImageFont, ImageDraw
from utils.datasets import *
from utils.plots import *
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import time
import csv
import lightTest
from utils.draw import draw_boxes, pil_draw_box_2
from licence import Licence
from utils.general import *
import ctypes

from utils.torch_utils import load_classifier


class YOLO(object):
    _defaults = {
        "model_path": 'yolo4_weights.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, weights):
        self.__dict__.update(self._defaults)
        self.dir = ['UP', 'RIGHT', 'DOWN', "LEFT"]
        self.currentCarID = 0
        self.virtureLine = [[0, 0], [0, 0]]
        self.carCnt = 0
        self.motoCnt = 0
        self.personCnt = 0
        self.busCnt = 0
        self.truckCnt = 0
        self.flag = False
        self.trafficLine = None
        self.trafficLight = None
        self.curpath = 0
        self.trafficLightColor = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(weights, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.licence = Licence()
        self.carDirection = {}
        self.config = None

    def detect_image(self, img, trafficline, path, idx_frame, illegal):

        global c
        self.trafficLine = trafficline
        self.curpath = path
        self.personCnt = self.carCnt = self.motoCnt = self.busCnt = self.truckCnt = 0
        im0 = img.copy()
        image = im0
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(
                torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()
        half = self.device.type != 'cpu'
        if half:
            self.model.half()
        img = letterbox(im0, new_shape=(640, 640))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, 0.45, 0.5)
        return_boxs = []
        return_class_names = []
        return_scores = []
        people_coords = []
        coords=[]
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:

                    c = cls
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2] - xyxy[0])
                    h = int(xyxy[3] - xyxy[1])
                    label = '%s' % (self.names[int(cls)])
                    if len(label) != 0:
                        self.flag = True
                        pass
                    if c == 9:
                        if self.flag:
                            self.trafficLight = xyxy
                            x1 = int(self.trafficLight[0])
                            y1 = int(self.trafficLight[1])
                            x2 = int(self.trafficLight[2])
                            y2 = int(self.trafficLight[3])

                            w = x2 - x1
                            h = y2 - y1
                            imgLight = im0[y1:y2, x1:x2]
                            if w < h:
                                imgLight = self.rotate_bound(imgLight, 90)
                            self.trafficLightColor = lightTest.detectImg(imgLight)
                            if self.trafficLightColor == 'green':
                                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor,
                                             color=(0, 255, 0), line_thickness=3, length=20, corner_color=(0, 255, 0))
                            elif self.trafficLightColor == 'red':
                                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor,
                                             color=(255, 0, 0), line_thickness=3, length=20, corner_color=(0, 255, 0))
                            elif self.trafficLightColor == 'yellow':
                                plot_one_box(self.trafficLight, im0, label=self.trafficLightColor,
                                             color=(255, 255, 0), line_thickness=3, length=20, corner_color=(0, 255, 0))
                            continue

                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3, length=20,
                                 corner_color=(0, 255, 0))
                    if c == 0:
                        plot_dots_on_people(xyxy, im0)
                        people_coords.append(xyxy)
                        distancing(people_coords,im0,dist_thres_lim=(250,350),line_thickness=3)
                        self.personCnt = self.personCnt + 1
                        if os.path.exists(self.curpath + "illegal/"):
                            pass
                        else:
                            os.mkdir(self.curpath + "illegal")
                        if os.path.exists(self.curpath + "illegal/runred"):
                            pass
                        else:
                            os.mkdir(self.curpath + "illegal/runred")

                        if self.trafficLine is not None:
                            if (self.trafficLightColor == 'green') and x >= self.trafficLine[0] and x + w <= \
                                    self.trafficLine[2] and (h / w >= 1.6):

                                if idx_frame % 6 == 0:
                                    imgTmp = im0[y:y + h, x:x + w]
                                    cv2.imwrite(self.curpath + "illegal/runred/" + str(idx_frame) + ".jpg", imgTmp)

                                    if isinstance(illegal.get(idx_frame, 0), int):
                                        illegal[idx_frame] = {}

                                    illegal[idx_frame].update({'runred': True})
                                font = ImageFont.truetype(font='model_data/simhei.ttf',
                                                          size=np.floor(0.012 * np.shape(im0)[1]).astype('int32'))
                                im0 = pil_draw_box_2(im0, [x, y, x + w, y + h], label="此人正在闯红灯", font=font)
                    if c == 2:
                        self.carCnt = self.carCnt + 1
                    if c == 3:
                        self.motoCnt = self.motoCnt + 1
                    if c == 5:
                        plot_dots_on_people(xyxy, im0)
                        coords.append(xyxy)
                        distance(im0,people_coords=people_coords, coords=coords,dist_thres_lim=(250, 350), line_thickness=3,
                                 label1='%s' % (self.names[int(0)]), label2='%s' % (self.names[int(c)]))
                        self.busCnt = self.busCnt + 1
                    if c == 7:
                        plot_dots_on_people(xyxy, im0)
                        coords.append(xyxy)
                        distance(im0,people_coords=people_coords, coords=coords, dist_thres_lim=(250,350), line_thickness=3,
                                 label1='%s' % (self.names[int(0)]), label2='%s' % (self.names[int(c)]))
                        self.truckCnt = self.truckCnt + 1
                    if c != 2:
                        continue
                    if y + h < im0.shape[0] - 12:
                        return_boxs.append([x + w / 2, y + h / 2, w, h])
                        return_class_names.append(self.names[int(cls)])
                        return_scores.append(conf)

        im0 = cv2.putText(im0, "moto:%d car:%d person:%d bus:%d truck:%d" % (
            self.motoCnt, self.carCnt, self.personCnt, self.busCnt, self.truckCnt),
                          (0, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)

        return np.array(return_boxs), np.array(return_scores), np.array(return_class_names), im0

    @staticmethod
    def rotate_bound(image, angle):

        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        return cv2.warpAffine(image, M, (nW, nH))
