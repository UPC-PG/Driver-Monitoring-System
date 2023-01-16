import cv2
import mediapipe as mp
import math
import dataPack

class HandsFreeDetection:
    def __init__(self):
        pass

    def vector_2d_angle(self, v1, v2):
        '''
            求解二维向量的角度
        '''
        v1_x=v1[0]
        v1_y=v1[1]
        v2_x=v2[0]
        v2_y=v2[1]
        try:
            angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
        except:
            angle_ =65535.
        if angle_ > 180.:
            angle_ = 65535.
        return angle_

    def h_gesture(self, angle_list):
        '''
            # 二维约束的方法定义手势
            # fist five gun love one six three thumbup yeah
        '''
        thr_angle = 65.
        thr_angle_thumb = 53.
        thr_angle_s = 49.
        gesture_str = None
        if 65535. not in angle_list:
            if (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (
                    angle_list[3] < thr_angle_s) and (angle_list[4] < thr_angle_s):
                gesture_str = "Warn"
        return gesture_str

    def hand_angle(self, hand_):
        '''
            获取对应手相关向量的二维角度,根据角度确定手势
        '''
        angle_list = []
        #---------------------------- thumb 大拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
            ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
            )
        angle_list.append(angle_)
        #---------------------------- index 食指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
            ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
            )
        angle_list.append(angle_)
        #---------------------------- middle 中指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
            ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
            )
        angle_list.append(angle_)
        #---------------------------- ring 无名指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
            ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
            )
        angle_list.append(angle_)
        #---------------------------- pink 小拇指角度
        angle_ = self.vector_2d_angle(
            ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
            ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
            )
        angle_list.append(angle_)
        return angle_list

    def detect(self, datapack):
        mp_drawing = mp.solutions.drawing_utils

        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        datapack.srcFrame = cv2.cvtColor(datapack.srcFrame, cv2.COLOR_BGR2RGB)
        results = hands.process(datapack.srcFrame)
        datapack.srcFrame = cv2.cvtColor(datapack.srcFrame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(datapack.dstFrame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # 绘制
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * datapack.srcFrame.shape[1]
                    y = hand_landmarks.landmark[i].y * datapack.srcFrame.shape[0]
                    hand_local.append((x, y))
                    # 记录能够识别得到的关键点的真实坐标

                if hand_local:
                    angle_list = self.hand_angle(hand_local)
                    # 各指头部分关键点之间的距离
                    gesture_str = self.h_gesture(angle_list)
                    cv2.putText(datapack.dstFrame, gesture_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                    if gesture_str == "Warn":
                        datapack.isFreeing = True
        return datapack

