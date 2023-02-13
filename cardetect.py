import time

import requests
import base64
from aip import AipImageClassify
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class detectCar(object):
    def __init__(self):
        pass

    def get_file_content(self,img):
        with open(img, 'rb') as fp:
            return fp.read()

    def detect(self,img):
        APP_ID = '25786029'
        API_KEY = 'PuEPkTUFRiGnrWxbjGOMCGKQ'
        SECRET_KEY = 'PZDrw5S4TiGk2orRpgGkqqNh7POzTbLt'
        client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)
        image = self.get_file_content(img)
        return client.carDetect(image, options={"top_num": 1}).get('result')[0]['name'],client.carDetect(image, options={"top_num": 1}).get('result')[0]['year'],client.carDetect(image, options={"top_num": 1})['color_result']


