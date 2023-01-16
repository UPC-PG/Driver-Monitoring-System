from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np  # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
import dataPack

def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear

class BlinkDetect:
    def __init__(self):
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.2
        # 闪烁阈值
        self.EYE_AR_CONSEC_FRAMES = 2
        # 初始化帧计数器和眨眼总数
        self.bCounter = 0
        self.bTotal = 0
        # 定义时间变量和判断标志
        self.t0 = self.t1 = time.time()
        self.flag = self.bool_sleep = 0
        self.k = 0
        # 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
        print("[眼睛检测]读取面部关键点")
        # 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 第三步：获取左右眼标志的索引
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def blinkDet(self, datapack):
        if self.flag == 0:
            self.t0 = time.time()
            self.flag = 1
        else:
            self.t1 = time.time()

        # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
        datapack.srcFrame = imutils.resize(datapack.srcFrame, width=720)
        datapack.dstFrame = imutils.resize(datapack.dstFrame, width=720)
        gray = cv2.cvtColor(datapack.srcFrame, cv2.COLOR_BGR2GRAY)
        # 第六步：使用detector(gray, 0) 进行脸部位置检测
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)

            # 第八步：将脸部特征信息转换为数组array的格式
            shape = face_utils.shape_to_np(shape)

            # 第九步：提取左眼和右眼坐标
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(datapack.dstFrame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(datapack.dstFrame, [rightEyeHull], -1, (0, 255, 255), 1)

            # 第十二步：进行画图操作，用矩形框标注人脸
            left = rect.left()
            top = rect.top()
            right = rect.right()
            bottom = rect.bottom()
            cv2.rectangle(datapack.dstFrame, (left, top), (right, bottom), (255, 0, 0), 1)

            '''
                分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
            '''
            # 第十三步：循环，满足条件的，眨眼次数+1
            if ear < self.EYE_AR_THRESH:  # 眼睛长宽比：0.2
                self.bCounter += 1
                self.k += 1
                if self.bCounter >= 20:  # 超过阈值一段时间后判定为闭眼
                    cv2.putText(datapack.dstFrame, "CLOSE!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    datapack.isBlinking = True
            else:
                # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                if self.bCounter >= self.EYE_AR_CONSEC_FRAMES:  # 阈值：2
                    self.bTotal += 1
                # 重置眼帧计数器
                self.bCounter = 0

            cv2.putText(datapack.dstFrame, "Blinks: {}".format(self.bTotal), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(datapack.dstFrame, "COUNTER: {}".format(self.bCounter), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(datapack.dstFrame, "K: {:.2f}".format(self.k), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 第十五步：进行画图操作，68个特征点标识
            for (x, y) in shape:
                cv2.circle(datapack.dstFrame, (x, y), 1, (0, 0, 255), -1)

        # 确定疲劳提示
        if 0 < self.t1 - self.t0 < 3 and self.bool_sleep == 1:
            cv2.putText(datapack.dstFrame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            datapack.isSleeping = True
        elif self.t1 - self.t0 > 3 and self.bool_sleep == 1:
            self.bool_sleep = 0
        elif self.t1 - self.t0 > 10:
            if self.k > 80:  # 10秒内有一定时间超过阈值（闭眼）即判定疲劳
                self.bool_sleep = 1
            self.flag = self.k = 0

        return datapack