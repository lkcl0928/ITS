"""
@ capture multiple cameras images to different folder
"""
import shutil

import cv2
import numpy as np
import time
import os
import threading
from time import ctime, sleep
import queue

abspath = os.getcwd()


class threadCameraRSTP(threading.Thread):
    """Hikvision camera
    @user   User name.
    @passwd User password.
    @ip     Camera ip name.
    @queue  Output queue.
    """

    def __init__(self, user, passwd, ip, queue):
        threading.Thread.__init__(self)
        self.user = user
        self.passwd = passwd
        self.ip = ip
        self.q = queue

    def run(self):
        access = "http://%s:%s@%s:8081" % (self.user, self.passwd, self.ip)
        cap = cv2.VideoCapture(access)
        if cap.isOpened():
            print('camera ' + self.ip + " connected.")

        while True:
            ret, img = cap.read()
            if ret:
                self.q.put(img)
                self.q.get() if self.q.qsize() > 2 else time.sleep(0.01)


class threadCameraUSB(threading.Thread):
    """usb camera
    @access   Usb descriptor.
    @queue    Output queue.
    """

    def __init__(self, access, queue):
        threading.Thread.__init__(self)
        self.access = access
        self.q = queue

    def run(self):
        cap = cv2.VideoCapture(self.access)
        if cap.isOpened():
            print('camera usb ' + str(self.access) + " connected.")

        while True:
            ret, img = cap.read()
            if ret:
                self.q.put(img)
                self.q.get() if self.q.qsize() > 2 else time.sleep(0.01)


def image_save(queueImage, queueCmd, dstDir, startNum, identification, display=True):
    """save image
    @queueImage   Image input queue.
    @queueCmd     Command input queue.
    @dstDir       Folder storing images.
    @startNum     Images number.
    @identification     Image show name.
    @display       Whether display image.
    """
    count = startNum
    if display:
        cv2.namedWindow(identification, 0)
    img = None

    while (True):
        if queueCmd.qsize() > 0:
            cmd = queueCmd.get()
            if cmd == 's':
                while True:
                    if queueImage.qsize() > 0:
                        img = queueImage.get()
                        cv2.imwrite(dstDir + '/' + ('%s' % count) + ".png", img)
                        #print(dstDir)
                        print(identification + 'save img' + str(count) + 'OK')
                        count += 1
                        break

        if queueImage.qsize() > 1:
            img = queueImage.get()
            if display:
                img0 = cv2.resize(img, (480, 640))
                cv2.resizeWindow(identification, 480, 640)
                cv2.imshow(identification, img0)

        cv2.waitKey(1)


def captureMutipleCamera(camera_access=[], start_Num=0, display=True):
    """save image
    @camera_access  All paremeters to capture and save the image, list, the format lile,
                    [
                        ["HIKVISION", "admin", "aaron20127", "192.168.0.111", 'D:/data/image'],
                        ["USB", 0, 'D:/data/image']
                    ]
    @start_Num     Image show name.
    @display       Whether display image.
    """

    # there must import queue, I don't know why
    import queue

    queueCmds = []
    thread_ids = []

    ## 1.start camera threads and save threads
    for camera in camera_access:
        identification = None
        cameraThread = None
        dstDir = None
        queueImage = queue.Queue(maxsize=4)
        queueCmd = queue.Queue(maxsize=4)

        if camera[0] == "HIKVISION":
            identification = camera[3]
            dstDir = camera[4]
            cameraThread = threadCameraRSTP(camera[1], camera[2], camera[3], queueImage)

        elif camera[0] == "USB":
            identification = "USB " + str(camera[1])
            dstDir = camera[2]
            cameraThread = threadCameraUSB(camera[1], queueImage)

        # camera thread
        thread_ids.append(cameraThread)

        # save image thread
        thread_ids.append(threading.Thread(target=image_save, args=(
            queueImage, queueCmd, dstDir, start_Num, identification, display)))

        # cmd input queue
        queueCmds.append(queueCmd)

    for thread in thread_ids:
        thread.daemon = True
        thread.start()

    ## 2. Enter enter to save image
    time.sleep(4)
    print("Please confirm the image saving mode:")
    print(" -- p (Press enter to save the image manually.)")
    print(" -- a (Automatically save pictures at intervals.)")

    mode = input()
    while (not (mode == 'p' or mode == 'a')):
        print("Input error, please input p or a.")
        mode = input()

    intervel = 5
    if mode == 'a':
        print('')
        print("Autosave mode !")
        print("Please input the intervals to autosave image times/seconds.")
        intervel = input()
        while (not intervel.isdigit()):
            print("")
            print("Please input a positive integer.")
            intervel = input()

        print("")
        print("Start auto saving ... n")
        intervals_time = 1.0 / int(intervel)
        start = time.time()
        while (True):
            if (time.time() - start) > intervals_time:
                for queue in queueCmds:
                    if queue.qsize() == 0:
                        queue.put('s')
                start = time.time()

    else:
        print('')
        print("Enter mode !")
        print("Please press 'Enter' to start saving ... n")

        while (True):
            if (input() == ''):
                for queue in queueCmds:
                    if queue.qsize() == 0:
                        queue.put('s')


if __name__ == '__main__':
    """example to capture camera
    """
    if os.path.exists(abspath + '\image\left'):
        shutil.rmtree(abspath + '\image\left')
    os.mkdir(abspath + '\image\left')
    if os.path.exists(abspath + '/image/usb'):
        shutil.rmtree(abspath + '/image/usb')
    os.mkdir(abspath + '/image/usb')
    if os.path.exists(abspath + '/image/usb1'):
        shutil.rmtree(abspath + '/image/usb1')
    os.mkdir(abspath + '/image/usb1')
    if os.path.exists(abspath + '/image/usb2'):
        shutil.rmtree(abspath + '/image/usb2')
    os.mkdir(abspath + '/image/usb2')
    camera_access = [
        ["HIKVISION", "admin", "admin", "192.168.0.101", abspath + '/image/left'],
        ["USB", "c003_mtmct_vis.mp4", abspath + '/image/usb'],
        ["USB", "c004_mtmct_vis.mp4", abspath + '/image/usb1'],
        ["USB", 0, abspath + '/image/usb2']
    ]

    captureMutipleCamera(camera_access, start_Num=0, display=True)
