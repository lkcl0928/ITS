import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageFont
from models.experimental import attempt_load
from utils.draw import draw_boxes
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box2
from dis_count import *
from utils.datasets import *
from deep_sort import build_tracker
from utils.parser import get_config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Det_Track_Dis(object):
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        cfg = get_config()
        cfg.merge_from_file('./configs/deep_sort.yaml')
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.device = torch.device('cpu')
        self.model = attempt_load('weights/yolov5s.pt', map_location=self.device)  # load FP32 model
        self.view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def main(self):
        while self.cap.grab():
            ret, frame = self.cap.retrieve()
            if ret:
                left_frame = frame[0:720, 0:1280]
                right_frame = frame[0:720, 1280:2560]
                im0 = left_frame.copy()
                img = letterbox(im0, new_shape=640, auto=True)[0]
                # Stack
                img = np.stack(img, 0)
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0   # torch.Size([1, 3, 480, 640])
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = self.model(img, augment=False)[0]
                # Apply NMS
                pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
                dislist, disp, deep = dis_co(left_frame, right_frame)
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        return_boxs = []
                        return_scores = []
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in det:
                            x = ((xyxy[2] - xyxy[0]) / 2) + xyxy[0]
                            y = ((xyxy[3] - xyxy[1]) / 2) + xyxy[1]
                            w = int(xyxy[2] - xyxy[0])
                            h = int(xyxy[3] - xyxy[1])
                            c=int(cls)
                            if x < 384:
                                pos = 'left'
                            elif (x < 896) and (x > 384):
                                pos = 'mid'
                            else:
                                pos = 'right'
                            x = int(x.cpu())
                            y = int(y.cpu())
                            dis = ((dislist[int(y), int(x), 0] ** 2 + dislist[int(y), int(x), 1] ** 2 + dislist[
                                 int(y), int(x), 2] ** 2) ** 0.5) / 100
                            label = '%s %.2f %.2fm %s' % (self.names[c], conf, dis, pos)
                            if c==0:
                                plot_one_box2(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                                cv2.circle(im0,(x,y),8,(0,0,255),-1)
                            if c!=0:
                                continue
                            return_boxs.append([x, y, w, h])
                            return_scores.append(conf)
                            outputs = self.deepsort.update(np.array(return_boxs), np.array(return_scores), im0)
                            if len(outputs) > 0:
                                bbox_xyxy = outputs[:, :4]
                                identities = outputs[:, -1]
                                img = draw_boxes(im0, bbox_xyxy, identities)
                    if self.view_img:
                        cv2.imshow('left', im0)
                        cv2.imshow('depth', deep)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    det=Det_Track_Dis()
    det.main()




