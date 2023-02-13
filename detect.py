import csv
import ctypes
import config
from cardetect import detectCar
from yolo import YOLO
import warnings
from deep_sort import build_tracker
from utils.draw import pil_draw_box_2, pil_draw_box3
from others.zebra_crossing import *
from utils.plots import *
from utils.general import *
from licence import Licence
import shutil
from pyecharts import Line, Pie
import json
from utils.parser import get_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoTracker(object):
    def __init__(self):
        self.carLabel = {}
        self.detectCar = detectCar()
        warnings.filterwarnings("ignore")
        self.yolo = 0
        self.video_path = 0
        self.path = 0
        self.outputDir = 0
        self.carLocation1 = {}
        self.carLocation2 = {}
        self.carSpeed = {}
        self.carDirection = {}
        self.carPre = {}
        self.virtureLine = [[0, 0], [0, 0]]
        self.carInCnt = 0
        self.carOutCnt = 0
        self.inCar = set()
        self.outCar = set()
        self.trafficLine = []
        self.idx_frame = 0
        self.carSpeedLimit = 0
        self.filename = 0
        self.trafficJamLimit = 0
        self.rett = False
        self.carLicense = {}
        self.speed = [0, 0, 0, 0, 0, 0, 0, 0]
        self.licence = Licence()
        self.frameAll = 0
        self.vdo = cv2.VideoCapture()
        self.vout = cv2.VideoWriter()
        self.saveVideoFlag = True
        self.displayFlag = True
        self.videoFps = 0
        self.videoHeight = 0
        self.trafficLine1 = []
        self.trafficLine2 = []
        self.laneline2 = [0, 0, 0, 0, 0, 0, 0, 0]
        self.carInfor = {}
        self.videoWidth = 0
        self.carSpeedLimit = 0
        self.trafficJamLimit = 0
        self.yolov5weight = 0
        self.carFromLeft = {}
        self.carFromRight = {}
        self.carFromForward = {}
        self.carTurn = {}
        self.keys=['车型','速度']
        self.endFlag = False
        self.illegal = {}
        self.trafficLightFlag = False
        self.trafficLightColor = None
        self.dir = ['UP', 'RIGHT', 'DOWN', "LEFT"]
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.dict_box = {}
        self.detector = self.yolo
        cfg = get_config()
        cfg.merge_from_file('./configs/deep_sort.yaml')
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    @staticmethod
    def color10():
        COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30),
                     (220, 20, 60),
                     (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237),
                     (138, 43, 226),
                     (238, 130, 238),
                     (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0),
                     (255, 239, 213),
                     (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222),
                     (65, 105, 225),
                     (173, 255, 47),
                     (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
                     (144, 238, 144),
                     (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
                     (128, 128, 128),
                     (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
                     (255, 245, 238),
                     (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255),
                     (176, 224, 230),
                     (0, 250, 154),
                     (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
                     (240, 128, 128),
                     (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
                     (255, 248, 220),
                     (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

        return COLORS_10[random.randint(3, 60) % len(COLORS_10)]

    @staticmethod
    def detectTrafficLine(img):
        Ni, Nj = (90, 1300)

        M = np.array([[-1.86073726e-01, -5.02678929e-01, 4.72322899e+02],
                      [-1.39150388e-02, -1.50260445e+00, 1.00507430e+03],
                      [-1.77785988e-05, -1.65517173e-03, 1.00000000e+00]])

        iM = inv(M)
        xy = np.zeros((640, 640, 2), dtype=np.float32)
        for py in range(640):
            for px in range(640):
                xy[py, px] = np.array([px, py], dtype=np.float32)
        ixy = cv2.perspectiveTransform(xy, iM)
        mpx, mpy = cv2.split(ixy)
        mapx, mapy = cv2.convertMaps(mpx, mpy, cv2.CV_16SC2)
        gray = preprocessing(img)
        canny = cv2.Canny(gray, 30, 90, apertureSize=3)
        Amplitude, theta = getGD(canny)
        indices, patches = zip(
            *sliding_window(Amplitude, theta, patch_size=(Ni, Nj)))  # use sliding_window get indices and patches
        labels = predict(patches, False)  # predict zebra crossing for every patches 1 is zc 0 is background
        indices = np.array(indices)
        ret, location = getlocation(indices, labels, Ni, Nj)
        return ret, location

    def calculateSpeed(self, location1, location2, cnt, flag=0):

        x11, y11, x12, y12 = location1
        x21, y21, x22, y22 = location2

        w1 = x12 - x11
        h1 = y12 - y11
        w2 = x22 - x21
        h2 = y22 - y21

        cx1, cy1 = x11 + w1 / 2, y11 + h1 / 2
        cx2, cy2 = x21 + w2 / 2, y21 + h2 / 2
        dis = math.sqrt(pow(abs(cx2 - cx1), 2) + pow(abs(cy2 - cy1), 2))
        h = (h1 + h2) / 2
        w = (w1 + w2) / 2
        if w1 / h1 >= 2 or w2 / h2 >= 2 or flag == 1:
            dpix = 1.8 / h
            dis = dis * dpix
            v = dis * 3.6 / cnt * self.videoFps
            return v

        dpix = 7.6 / w
        dis = dis * dpix
        v = dis * 3.6 / cnt * self.videoFps
        return v

    @staticmethod
    def calculateDirection(location1, location2):
        x11, y11, x12, y12 = location1
        x21, y21, x22, y22 = location2

        w1 = x12 - x11
        h1 = y12 - y11
        w2 = x22 - x21
        h2 = y22 - y21

        cx1, cy1 = x11 + w1 / 2, y11 + h1 / 2
        cx2, cy2 = x21 + w2 / 2, y21 + h2 / 2
        dx = cx2 - cx1
        dy = cy2 - cy1
        if dy > 0 and 3 * abs(dy) >= abs(dx):
            return 2
        if dy < 0 and 1.5 * abs(dy) >= abs(dx):
            return 0
        if dx > 0 and abs(dx) >= abs(dy):
            return 1
        if dx < 0 and abs(dx) >= abs(dy):
            return 3

        return 0

    def saveVideo(self):

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = self.vdo.get(cv2.CAP_PROP_FPS)

        size = (int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.vout = cv2.VideoWriter(self.outputDir + self.filename + "_processed.mp4", fourcc, fps, size)

    def run(self):
        global x_center, y_center, ori_im, font
        self.carSpeedLimit = config.CARSPEEDLIMIT
        self.trafficJamLimit = config.TRAFFICJAMLIMIT
        self.yolov5weight = config.YOLOWEIGHT
        self.detector = YOLO(self.yolov5weight)
        self.path = config.PATH
        self.video_path = self.path
        filepath, filename = os.path.split(self.path)
        self.outputDir = os.getcwd() + "/output/" + filename.split('.')[0] + "/"
        self.filename = filename.split('.')[0]

        if os.path.exists(self.outputDir):
            shutil.rmtree(self.outputDir)
        os.mkdir(self.outputDir)
        self.vdo = cv2.VideoCapture(self.video_path)
        self.videoWidth, self.videoHeight = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.videoFps = self.vdo.get(cv2.CAP_PROP_FPS)
        self.frameAll = self.vdo.get(7)
        results = []
        self.idx_frame = 0
        self.saveVideo()
        self.trafficLine1 = config.trafficline1
        while self.vdo.isOpened():
            self.idx_frame += 1
            start = time.time()
            _, ori_im = self.vdo.read()
            if not _:
                raise RuntimeError("Can't receive frame (stream end?). Exiting ...")
            image = ori_im
            if len(self.trafficLine1) > 0:
                self.trafficLine = self.trafficLine1
                pass
            else:
                if self.idx_frame < 5 and self.rett == False:
                    ret, location = self.detectTrafficLine(image)
                    if ret != 0:
                        self.rett = True
                        self.trafficLine = [location[0][0], location[0][1], location[1][0], location[1][1]]

            self.virtureLine = [0, int(image.shape[0] / 2 - 90), int(image.shape[1]), int(image.shape[0] / 2 - 90)]
            cv2.line(image, (0, int(image.shape[0] / 2 - 90)), (int(image.shape[1]), int(image.shape[0] / 2 - 90)),
                     color=(255, 0, 0), thickness=6)
            im1 = ori_im.copy()
            im = im1

            bbox_xywh, cls_conf, cls_ids, ori_im = self.detector.detect_image(ori_im, self.trafficLine, self.outputDir,
                                                                              self.idx_frame, self.illegal)
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]

                for i, bbox in enumerate(bbox_xyxy):
                    x1, y1, x2, y2 = bbox_xyxy[i]
                    box_xywh = xyxy2tlwh(bbox_xyxy)
                    for j in range(len(box_xywh)):
                        x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                        y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标

                        ids = outputs[j][-1]
                        center = [x_center, y_center]
                        self.dict_box.setdefault(ids, []).append(center)
                    if self.idx_frame > 2:
                        for key, value in self.dict_box.items():
                            for a in range(len(value) - 1):
                                color = self.color10()
                                index_start = a
                                index_end = index_start + 1
                                cv2.circle(ori_im, (int(x_center), int(y_center)), 3, color, 5, cv2.LINE_AA)
                                cv2.line(ori_im, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                                         # map(int,"1234")转换为list[1,2,3,4]
                                         color, 5, cv2.LINE_AA)

                    id = int(identities[i]) if identities is not None else 0
                    if self.carLicense.get(id, 0) == 0:
                        t = self.licence.detectLicence(ori_im[y1:y2, x1:x2], x1, y1)
                        if t is not None:
                            xyxy, label = t
                            self.carLicense[id] = label
                    if self.carLabel.get(id, 0) == 0 and config.CARCLASSIFY:
                        t = self.detectCar.detect((ori_im[y1:y2, x1:x2]))
                        if t is not None:
                            label = t
                            self.carLabel[id] = label

                    if self.carPre.get(id, 0) == 0:
                        self.carPre[id] = self.idx_frame
                        self.carLocation1[id] = [x1, y1, x2, y2]
                    elif (self.idx_frame - self.carPre[id]) >= 4:

                        self.carLocation2[id] = [x1, y1, x2, y2]
                        if self.carLocation1[id][3] <= self.virtureLine[1] <= y2:
                            self.inCar.add(id)
                        elif self.carLocation1[id][1] >= self.virtureLine[1] >= y1:
                            self.outCar.add(id)

                        pre = self.carDirection.get(id, 0)
                        self.carSpeed[id] = self.calculateSpeed(self.carLocation1[id], self.carLocation2[id],
                                                                self.idx_frame - self.carPre[id])
                        if self.carSpeed[id] >= 8:
                            self.carDirection[id] = self.dir[
                                self.calculateDirection(self.carLocation1[id], self.carLocation2[id])]
                        if self.carDirection.get(id, 0) == 'LEFT' or self.carDirection.get(id, 0) == 'RIGHT':
                            self.carSpeed[id] = self.calculateSpeed(self.carLocation1[id], self.carLocation2[id],
                                                                    self.idx_frame - self.carPre[id], 1)
                        if self.carSpeed.get(id, 0) > self.carSpeedLimit:
                            if os.path.exists(self.outputDir + "illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal")
                            if os.path.exists(self.outputDir + "illegal/overspeed"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal/overspeed")

                            imgTmp = ori_im[y1:y2, x1:x2]
                            imgg = imgTmp.copy()
                            imgg = Image.fromarray(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
                            if isinstance(self.illegal.get(self.idx_frame, 0), int):
                                self.illegal[self.idx_frame] = {}

                            self.illegal[self.idx_frame].update(
                                {'overspeed': str(id) + " " + self.carLicense.get(id, "")})
                            imgg.save(
                                self.outputDir + "illegal/overspeed/" + str(self.idx_frame) + "_" + self.carLicense.get(
                                    id, "") + ".jpg")
                            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                                      size=np.floor(0.012 * np.shape(ori_im)[1]).astype('int32'))
                            ori_im = pil_draw_box_2(ori_im, [x1, y1, x2, y2], label="超速", font=font)
                            player = ctypes.windll.kernel32
                            for i in range(3):
                                player.Beep(800, 1000)
                        if pre == 'UP':
                            if self.carDirection.get(id, 0) == 'LEFT' and self.carSpeed.get(id, 0) >= 8:
                                self.carTurn[id] = 'LEFT'
                            elif self.carDirection.get(id, 0) == 'RIGHT' and self.carSpeed.get(id, 0) >= 8:
                                self.carTurn[id] = 'RIGHT'

                        if self.carTurn.get(id, 0) == 'LEFT' and (self.carFromRight.get(id, 0) == True or self.carFromForward.
                                get(id, 0) == True) and self.carFromLeft.get(id, 0) != True:
                            ori_im = pil_draw_box_2(ori_im, [x1, y1, x2, y2], label="未按导向行驶", font=font)
                            # player = ctypes.windll.kernel32
                            # for i in range(3):
                            # player.Beep(800, 1000)
                            if isinstance(self.illegal.get(self.idx_frame, 0), int):
                                self.illegal[self.idx_frame] = {}
                            self.illegal[self.idx_frame].update(
                                {'turnwrong': str(id) + " " + self.carLicense.get(id, "")})
                            if os.path.exists(self.outputDir + "illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal")
                            if os.path.exists(self.outputDir + "illegal/turnwrong"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal/turnwrong")
                            imgTmp = ori_im[y1:y2, x1:x2]
                            imgg = Image.fromarray(cv2.cvtColor(imgTmp, cv2.COLOR_BGR2RGB))
                            imgg.save(
                                self.outputDir + "illegal/turnwrong/" + str(self.idx_frame) + "_" + self.carLicense.get(
                                    id, "") + ".jpg")

                        elif self.carTurn.get(id, 0) == 'RIGHT' and (
                                self.carFromLeft.get(id, 0) == True or self.carFromForward.get(id,
                                                                                               0) == True) and self.carFromRight.get(
                            id, 0) != True:

                            ori_im = pil_draw_box_2(ori_im, [x1, y1, x2, y2], label="未按导向行驶", font=font)
                            # player = ctypes.windll.kernel32
                            # for i in range(3):
                            # player.Beep(800, 1000)
                            if isinstance(self.illegal.get(self.idx_frame, 0), int):
                                self.illegal[self.idx_frame] = {}
                            self.illegal[self.idx_frame].update(
                                {'turnwrong': str(id) + " " + self.carLicense.get(id, "")})
                            if os.path.exists(self.outputDir + "illegal/"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal")
                            if os.path.exists(self.outputDir + "illegal/turnwrong"):
                                pass
                            else:
                                os.mkdir(self.outputDir + "illegal/turnwrong")
                            imgTmp = ori_im[y1:y2, x1:x2]
                            imgg = Image.fromarray(cv2.cvtColor(imgTmp, cv2.COLOR_BGR2RGB))
                            imgg.save(
                                self.outputDir + "illegal/turnwrong/" + str(self.idx_frame) + "_" + self.carLicense.get(
                                    id, "") + ".jpg")

                        self.carLocation1[id][0] = self.carLocation2[id][0]
                        self.carLocation1[id][1] = self.carLocation2[id][1]
                        self.carLocation1[id][2] = self.carLocation2[id][2]
                        self.carLocation1[id][3] = self.carLocation2[id][3]
                        self.carPre[id] = self.idx_frame
                    # print(x1,x2,y1,y2)
                    if self.carDirection.get(id, 0) == 'UP' and (self.detector.trafficLightColor == 'green'):
                        pass
                    if self.carDirection.get(id, 0) == 'UP' and (
                            self.detector.trafficLightColor == 'red') and self.carSpeed.get(id, 0) == 0:
                        if len(config.laneline) > 0:
                            # print(config.laneline)
                            t = bbox_xyxy[i]
                            x1, y1, x2, y2 = t
                            # print(t)
                            tmp = t[:]
                            if self.judge_line_illegal(id, tmp, config):
                                # print(t)
                                # print(tmp)
                                x, y, xx, yy = tmp
                                ori_im = pil_draw_box_2(ori_im, tmp, label="车辆压停车线", font=font)
                                # player = ctypes.windll.kernel32
                                # for i in range(3):
                                # player.Beep(800, 1000)
                                if isinstance(self.illegal.get(self.idx_frame, 0), int):
                                    self.illegal[self.idx_frame] = {}
                                self.illegal[self.idx_frame].update(
                                    {'touchline': str(id) + " " + self.carLicense.get(id, "")})
                                if os.path.exists(self.outputDir + "illegal/"):
                                    pass
                                else:
                                    os.mkdir(self.outputDir + "illegal")
                                if os.path.exists(self.outputDir + "illegal/touchline"):
                                    pass
                                else:
                                    os.mkdir(self.outputDir + "illegal/touchline")
                                # print(x,y,xx,yy)
                                imgTmp = ori_im[y:yy, x:xx]
                                imgg = Image.fromarray(cv2.cvtColor(imgTmp, cv2.COLOR_BGR2RGB))
                                imgg.save(self.outputDir + "illegal/touchline/" + str(
                                    self.idx_frame) + "_" + self.carLicense.get(id, "") + ".jpg")

                    if self.carDirection.get(id, 0) == 'UP':
                        if len(config.laneline2) > 0:
                            # print(config.laneline2)
                            t = bbox_xyxy[i]
                            x1, y1, x2, y2 = t
                            tmp = t[:]
                            box_xywh = xyxy2tlwh(bbox_xyxy)
                            for j in range(len(box_xywh)):
                                x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                                y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
                                if self.laneline2[0] <= x_center <= self.laneline2[4]:
                                    # print(tmp)
                                    x, y, xx, yy = tmp
                                    ori_im = pil_draw_box_2(ori_im, tmp, label="车辆压双黄线", font=font)
                                    # player = ctypes.windll.kernel32
                                    # for i in range(3):
                                    # player.Beep(800, 1000)

                                    if isinstance(self.illegal.get(self.idx_frame, 0), int):
                                        self.illegal[self.idx_frame] = {}
                                    self.illegal[self.idx_frame].update({'touchline2': str(
                                        id) + " " + self.carLicense.get(id, "")})
                                    if os.path.exists(self.outputDir + "illegal/"):
                                        pass
                                    else:
                                        os.mkdir(self.outputDir + "illegal")
                                    if os.path.exists(self.outputDir + "illegal/touchline2"):
                                        pass
                                    else:
                                        os.mkdir(self.outputDir + "illegal/touchline2")
                                    # print(x, y, xx, yy)
                                    imgTmp = ori_im[y:yy, x:xx]
                                    imgg = Image.fromarray(cv2.cvtColor(imgTmp, cv2.COLOR_BGR2RGB))
                                    imgg.save(self.outputDir + "illegal/touchline2/" + str(
                                        self.idx_frame) + "_" + self.carLicense.get(id, "") + ".jpg")

                    self.carrunred(id, x1, x2, y1, y2, ori_im)
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + w / 2
                    if len(config.leftline):
                        # print(config.leftline)
                        if self.inArea(cx, cy, config.leftline):
                            self.carFromLeft[id] = True
                    if len(config.rightline):
                        # print(config.rightline)
                        if self.inArea(cx, cy, config.rightline):
                            self.carFromRight[id] = True
                    if len(config.forwardline):
                        # print(config.forwardline)
                        if self.inArea(cx, cy, config.forwardline):
                            self.carFromForward[id] = True

                    self.classify_speed(id)

                    self.carInfor[id] = self.carDirection.get(id, '') + "," + self.carLicense.get(id, '') + "," + str(
                        int(self.carSpeed.get(id, 0)))
                font = ImageFont.truetype(font='model_data/simhei.ttf',
                                          size=np.floor(0.012 * np.shape(ori_im)[1]).astype('int32'))
                ori_im = pil_draw_box3(ori_im, bbox_xyxy, identities, self.carSpeed, self.carLicense, self.carDirection,
                                      font)

                if self.detector.carCnt + self.detector.busCnt + self.detector.truckCnt >= self.trafficJamLimit and self.idx_frame % 8 == 0:
                    if isinstance(self.illegal.get(self.idx_frame, 0), int):
                        self.illegal[self.idx_frame] = {}
                    self.illegal[self.idx_frame].update({'trafficjam': True})
                    cv2img = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                    pilimg = Image.fromarray(cv2img)
                    draw = ImageDraw.Draw(pilimg)
                    font = ImageFont.truetype("./simhei.ttf", 100, encoding="utf-8")
                    draw.text((1350, 20), '交通拥堵!!!', (255, 0, 0), font=font)
                    player = ctypes.windll.kernel32
                    for i in range(3):
                        player.Beep(800, 1000)
                    ori_im = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
            end = time.time()
            if len(self.trafficLine1) > 0:
                plot_one_box(self.trafficLine1, ori_im, label="Zebra crossing1", color=(105, 210, 255),
                             line_thickness=3, length=20, corner_color=(0, 255, 0))
            elif (self.rett):
                label = "Zebra crossing"
                self.trafficLine[0] = min(self.videoWidth / 6, self.trafficLine[0])
                self.trafficLine[2] = max(self.videoWidth / 6 * 5, self.trafficLine[2])
                plot_one_box2(self.trafficLine, ori_im, label=label, color=(105, 210, 255), line_thickness=3)

            if len(config.leftline) > 0:
                # print(config.leftline)
                self.plot_lane(config.leftline, ori_im, "Leftline")
            if len(config.rightline) > 0:
                # print(config.rightline)
                self.plot_lane(config.rightline, ori_im, "Rightline")
            if len(config.forwardline) > 0:
                # print(config.forwardline)
                self.plot_lane(config.forwardline, ori_im, "Forwardline")
            if len(config.laneline2) > 0:
                # print(config.laneline2)
                self.plot_lane(config.laneline2, ori_im, "Laneline2")
            cv2.putText(ori_im, "carIn:%d carOut:%d" % (len(self.inCar), len(self.outCar)), (0, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            if config.SAVEVIDEO:
                self.vout.write(ori_im)
            self.generate_chart()
            self.save_csv_data()
            print('frame%d frameAll:%d' % (self.idx_frame, self.frameAll))

            cv2.imshow(str(self.video_path), ori_im)
            cv2.waitKey(1)
            if self.idx_frame==config.idx_frame:
                break
        self.vdo.release()
        self.vout.release()
        cv2.destroyAllWindows()


    def plot_lane(self, area, im, label=""):
        color = self.color10()
        x1, y1, x2, y2, x3, y3, x4, y4 = area
        cv2.putText(im, label, (x1 - 50, y1), 0, 3, color, thickness=5, lineType=cv2.LINE_4)
        cv2.line(im, (x1, y1), (x2, y2), color, 5)
        cv2.line(im, (x2, y2), (x4, y4), color, 5)
        cv2.line(im, (x4, y4), (x3, y3), color, 5)
        cv2.line(im, (x3, y3), (x1, y1), color, 5)

    def classify_speed(self, id):
        carID = id
        if (self.carSpeed.get(carID, 0)) >= 5 and (self.carSpeed.get(carID, 0)) < 10:
            self.speed[0] += 1
        elif (self.carSpeed.get(carID, 0)) >= 10 and (self.carSpeed.get(carID, 0)) < 25:
            self.speed[1] += 1
        elif (self.carSpeed.get(carID, 0)) >= 15 and (self.carSpeed.get(carID, 0)) < 20:
            self.speed[2] += 1
        elif (self.carSpeed.get(carID, 0)) >= 20 and (self.carSpeed.get(carID, 0)) < 25:
            self.speed[3] += 1
        elif (self.carSpeed.get(carID, 0)) >= 25 and (self.carSpeed.get(carID, 0)) < 30:
            self.speed[4] += 1
        elif (self.carSpeed.get(carID, 0)) >= 30 and (self.carSpeed.get(carID, 0)) < 35:
            self.speed[5] += 1
        elif (self.carSpeed.get(carID, 0)) >= 35 and (self.carSpeed.get(carID, 0)) < 40:
            self.speed[6] += 1
        elif (self.carSpeed.get(carID, 0)) >= 40:
            self.speed[7] += 1

    def carrunred(self, id, x1, x2, y1, y2, ori_im):

        if (self.detector.trafficLightColor == 'red') and self.carDirection.get(id, 0) == 'UP' and \
                self.carSpeed.get(id, 0) >= 20:

            if not self.carFromForward.get(id, False):
                return
            if y1 < self.trafficLine[3] - 10:
                pass
            if os.path.exists(self.outputDir + "illegal/"):
                pass
            else:
                os.mkdir(self.outputDir + "illegal")
            if os.path.exists(self.outputDir + "illegal/carrunred"):
                pass
            else:
                os.mkdir(self.outputDir + "illegal/carrunred")

            imgTmp = ori_im[y1:y2, x1:x2]
            self.illegal[self.idx_frame] = {}
            self.illegal[self.idx_frame].update({'carrunred': str(id) + " " + self.carLicense.get(id, "")})
            imgg = imgTmp.copy()

            imgg = Image.fromarray(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
            imgg.save(self.outputDir + "illegal/carrunred/" + str(self.idx_frame) + "_" +
                      self.carLicense.get(id, "") + ".jpg")

            font = ImageFont.truetype(font='model_data/simhei.ttf',
                                      size=np.floor(0.012 * np.shape(ori_im)[1]).astype('int32'))
            ori_im = pil_draw_box_2(ori_im, [x1, y1, x2, y2], label="车辆闯红灯", font=font)
            # player = ctypes.windll.kernel32
            # for i in range(3):
            # player.Beep(800, 1000)

    def save_csv_data(self):
        if os.path.exists(self.outputDir + "csv/"):
            pass
        else:
            os.mkdir(self.outputDir + "csv/")
        if self.idx_frame % 6 == 0:
            with open(self.outputDir + "csv/count.csv", 'a',newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow([int(self.idx_frame/6),int(self.detector.carCnt),
                                 int(self.detector.personCnt),int(self.detector.motoCnt),int(self.detector.busCnt),
                                 int(self.detector.truckCnt)])
            with open(self.outputDir + "csv/flow.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([int(self.idx_frame / 6), int(len(self.inCar)), int(len(self.outCar))])
            with open(self.outputDir + "csv/speed.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.speed)

    def judge_line_illegal(self, id, car, config):
        arr = []
        for i in range(len(config.laneline)):
            arr.append(config.laneline[i])
            if (i + 1) % 4 == 0:
                x1, y1, x2, y2 = arr
                arr = []
                q1 = [x1, y1]
                q2 = [x2, y2]
                tx1, ty1, tx2, ty2 = car
                p1 = [tx1, ty2]
                p2 = [tx2, ty2]

                if self.isIntersect(p1, p2, q1, q2):
                    return True
        return False

    @staticmethod
    def det(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return x1 * y2 - y1 * x2

    @staticmethod
    def dcmp(a):
        return abs(a) < 1e-6

    @staticmethod
    def dot(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return x1 * x2 + y1 * y2

    def on_seg(self, p1, p2, q):
        d1 = [p1[0] - q[0], p1[1] - q[1]]
        d2 = [p2[0] - q[0], p2[1] - q[1]]
        return self.dcmp(self.det(d1, d2)) and self.dot(d1, d2) <= 1e-6

    def intersection(self, p1, p2, q1, q2):
        p2p1 = [p2[0] - p1[0], p2[1] - p1[1]]
        q2q1 = [q2[0] - q1[0], q2[1] - q1[1]]
        q1p1 = [q1[0] - p1[0], q1[1] - p1[1]]
        t = self.det(q2q1, q1p1) / self.det(q2q1, p2p1)
        t2 = [p2p1[0] * t, p2p1[1] * t]
        res = [p1[0] + t2[0], p1[1] + t2[1]]
        return res

    def isIntersect(self, p1, p2, q1, q2):
        t = self.intersection(p1, p2, q1, q2)
        return self.on_seg(p1, p2, t) and self.on_seg(q1, q2, t)

    def generate_chart(self):
        if os.path.exists(self.outputDir + "chart/"):
            pass
        else:
            os.mkdir(self.outputDir + "chart/")
        line2 = Line("车辆流量分析图")
        t2 = ['time', 'CarIn', 'CarOut']
        if os.path.exists(self.outputDir + "csv/flow.csv"):
            with open(self.outputDir + "csv/flow.csv", 'r') as f:
                reader = csv.reader(f)
                result = np.array(list(reader))
                for i in range(len(result[0])):
                    if i == 0:
                        continue
                    line2.add(t2[i], result[0:, 0], result[0:, i])
                line2.render(self.outputDir + "chart/flow.html")
        pie = Pie("速度区间分析图")
        if os.path.exists(self.outputDir + "csv/speed.csv"):
            with open(self.outputDir + "csv/speed.csv", 'r') as f:
                reader = csv.reader(f)
                result = np.array(list(reader))
                pie.add("速度区间",
                        ["[5km/h,10km/h)", "[10km/h,15km/h)", "[15km/h,20km/h)", "[20km/h,25km/h)", "[25km/h,30km/h)",
                         "[30km/h,35 km/h)", "[35km/h,40 km/h)", "[40km/h,+infinity)"], result[-1])
                pie.render(self.outputDir + "chart/speed.html")
        self.saveIllegal()
        # break

    def saveIllegal(self):
        if os.path.exists(self.outputDir + "illegal/"):
            pass
        else:
            os.mkdir(self.outputDir + "illegal")
        with open(self.outputDir + "illegal/illegal.json", 'w') as f:
            json.dump(self.illegal, f)

    @staticmethod
    def inArea(x, y, area):
        x1, y1, x2, y2, x3, y3, x4, y4 = area
        if (x1 + x2) / 2 < x < (x3 + x4) / 2 and y1 < y < y2:
            return True
        return False

    def endDetect(self):
        self.endFlag = True


if __name__ == '__main__':
    videoTracker = VideoTracker()
    videoTracker.run()
