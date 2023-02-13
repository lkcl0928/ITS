import cv2
from utils.datasets import *
from utils.plots import *
from utils.general import *
from utils.torch_utils import load_classifier
import collections
import lightTest


class YOLO(object):
    def __init__(self):
        self.trafficLightColor = 0
        self.trafficLight = 0
        weights_path = 'weights/yolov5s.pt'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(weights_path, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    def detect_image(self, img):
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
        pred = non_max_suppression(pred, 0.35, 0.5)
        return_boxs = []
        return_class_names = []
        return_scores = []
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:

                    c = int(cls)
                    x = int(xyxy[0])
                    y = int(xyxy[1])
                    w = int(xyxy[2] - xyxy[0])
                    h = int(xyxy[3] - xyxy[1])
                    if c == 9:
                        self.trafficLight = xyxy
                        x1 = int(self.trafficLight[0])
                        y1 = int(self.trafficLight[1])
                        x2 = int(self.trafficLight[2])
                        y2 = int(self.trafficLight[3])
                        w = x2 - x1
                        h = y2 - y1
                        imgLight = im0[y1:y2, x1:x2]
                        if w > h:
                            imgLight = rotate_bound(imgLight, 90)
                        self.trafficLightColor = lightTest.detectImg(imgLight)
                        if self.trafficLightColor == 'green':
                            plot_one_box2(self.trafficLight, im0, label=f'{self.trafficLightColor} {conf:.2f}',
                                          color=(0, 255, 0), line_thickness=3)
                        elif self.trafficLightColor == 'red':
                            plot_one_box2(self.trafficLight, im0, label=f'{self.trafficLightColor} {conf:.2f}',
                                          color=(0, 0, 255), line_thickness=3)
                        elif self.trafficLightColor == 'yellow':
                            plot_one_box2(self.trafficLight, im0, label=f'{self.trafficLightColor} {conf:.2f}',
                                          color=(0, 255, 255), line_thickness=3)
                        continue
                    if c==0:
                        label1=f'Pedestrian {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label1, color=(78,90,155), line_thickness=3, length=20,
                                     corner_color=(0, 255, 0))
                    if c==2 or c==5 or c==6 or c== 7:
                        label2 = f'Vehicle {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label2, color=(125,34,56), line_thickness=3, length=20,
                                     corner_color=(0, 255, 0))
                    if c ==1 or c==3:
                        label3=f'Cyclist {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label3, color=(34,150,48), line_thickness=3, length=20,
                                     corner_color=(0, 255, 0))
                    if c != 2:
                        continue
                    return_boxs.append([x + w / 2, y + h / 2, w, h])
                    return_class_names.append(self.names[int(cls)])
                    return_scores.append(conf)
        return np.array(return_boxs), np.array(return_scores), np.array(return_class_names), im0


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

