import warnings
from yolov5 import YOLO
from deep_sort import build_tracker
from utils.draw import pil_draw_box
from utils.plots import *
from utils.general import *
from utils.parser import get_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoTracker(object):
    def __init__(self):
        warnings.filterwarnings("ignore")
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.detector = YOLO()
        cfg = get_config()
        cfg.merge_from_file('./configs/deep_sort.yaml')
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.vdo = cv2.VideoCapture()
        self.frameAll = 0
        self.idx_frame = 0
        self.dict_box = {}

    def run(self, video_path):
        global x_center, y_center
        self.vdo = cv2.VideoCapture(video_path)
        self.frameAll = self.vdo.get(7)
        self.idx_frame = 0
        videoWidth, videoHeight = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.vdo.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # size = (videoWidth, videoHeight)
        vout = cv2.VideoWriter("processed.mp4", fourcc, fps, (videoWidth, videoHeight))
        while self.vdo.isOpened():
            self.idx_frame += 1
            ret, img = self.vdo.read()
            if ret == True:
                im0 = img.copy()
                image = im0
                bbox_xywh, cls_conf, cls_ids, img = self.detector.detect_image(img)
                outputs = self.deepsort.update(bbox_xywh, cls_conf, image)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    font = ImageFont.truetype(font='model_data/simhei.ttf',
                                              size=np.floor(0.012 * np.shape(img)[1]).astype('int32'))
                    img = pil_draw_box(img, bbox_xyxy, identities, font)
                cv2.imshow(str(video_path), img)
                vout.write(img)
                print('frame%d frameAll:%d' % (self.idx_frame, self.frameAll))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.vdo.release()
        vout.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    vt = VideoTracker()
    vt.run('your demo path')
