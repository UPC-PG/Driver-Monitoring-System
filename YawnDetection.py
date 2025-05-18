import imutils
import time
import dlib
import cv2
from imutils import face_utils
import numpy as np
import dataPack

def mouth_aspect_ratio(mouth):
    A1 = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B1 = np.linalg.norm(mouth[4] - mouth[8])  # 53, 57
    C1 = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    K1 = (A1 + B1) / (2.0 * C1)
    A2 = np.linalg.norm(mouth[13] - mouth[19])  # 62, 68
    B2 = np.linalg.norm(mouth[15] - mouth[17])  # 64, 66
    C2 = np.linalg.norm(mouth[12] - mouth[16])  # 61, 65
    K2 = (A2 + B2) / (2.0 * C2)
    mar = (K1 + K2) / 2.0
    return mar


class YawnDetect:
    def __init__(self):
        # 打哈欠长宽比
        self.MAR_THRESH = 0.9
        # 闪烁阈值
        self.MOUTH_AR_CONSEC_FRAMES = 5
        # 哈欠次数
        self.mCounter = 0
        self.mTotal = 0
        # 初始时间和判断标志（****）
        self.t0 = self.t1 = time.time()
        self.flag = self.bool_sleep = 0
        self.k = 0
        # 初始化DLIB的人脸检测器（HOG）并创建面部标志物预测
        print("[哈欠检测]读取面部关键点")
        # 获得脸部位置检测器
        self.detector = dlib.get_frontal_face_detector()
        # 获得脸部特征位置检测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 分别获取左右眼面部标志的索引
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # srcFrame 为原视频帧, dstFrame 为绘图视频帧
    def yawnDet(self, datapack):
        if self.flag:
            self.t0 = time.time()
            self.flag = 1
        else:
            self.t1 = time.time()

            # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
            # ret, frame = cap.read()
            datapack.srcFrame = imutils.resize(datapack.srcFrame, width=720)
            datapack.dstFrame = imutils.resize(datapack.dstFrame, width=720)
            gray = cv2.cvtColor(
                datapack.srcFrame, 
                cv2.COLOR_BGR2GRAY)
            # 第六步：使用detector(gray, 0) 进行脸部位置检测
            rects = self.detector(gray, 0)

            # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
            for rect in rects:
                shape = self.predictor(gray, rect)

                # 第八步：将脸部特征信息转换为数组array的格式
                shape = face_utils.shape_to_np(shape)

                # 第九步：提取嘴巴坐标
                mouth = shape[self.mStart:self.mEnd]

                # 第十步：构造函数计算嘴部的MAR值
                mar = mouth_aspect_ratio(mouth)

                # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(datapack.dstFrame, [mouthHull], -1, (0, 255, 255), 1)

                # 第十二步：进行画图操作，用矩形框标注人脸
                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                cv2.rectangle(datapack.dstFrame, (left, top), (right, bottom), (255, 0, 0), 1)

                '''
                            计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
                '''
                # 同理，判断是否打哈欠
                if mar > self.MAR_THRESH:  # 张嘴阈值0.5
                    self.mCounter += 1
                    self.k += 1
                    cv2.putText(datapack.dstFrame, "Yawning!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 如果连续3次都小于阈值，则表示打了一次哈欠
                    if self.mCounter >= self.MOUTH_AR_CONSEC_FRAMES:  # 阈值：5
                        self.mTotal += 1
                        datapack.isYawning = True
                    # 重置嘴帧计数器
                    self.mCounter = 0
                cv2.putText(datapack.dstFrame, "Yawning: {}".format(self.mTotal), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                cv2.putText(datapack.dstFrame, "mCounter: {}".format(self.mCounter), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(datapack.dstFrame, "MAR: {:.2f}".format(mar), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 第十五步：进行画图操作，68个特征点标识
                for (x, y) in shape:
                    cv2.circle(datapack.dstFrame, (x, y), 1, (0, 0, 255), -1)

        # 确定疲劳提示
        if 0 < self.t1 - self.t0 < 3 and self.bool_sleep == 1:
            cv2.putText(datapack.dstFrame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif self.t1 - self.t0 > 3 and self.bool_sleep == 1:
            self.bool_sleep = 0
        elif self.t1 - self.t0 > 20:
            if self.k > 288:  # 20秒内有12秒超过阈值（打哈欠）即判定为疲劳
                self.bool_sleep = 1
            self.flag = self.k = 0

        return datapack



