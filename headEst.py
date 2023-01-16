import cv2
import numpy as np
import paddlehub as hub
from paddlehub.common.logger import logger
import time
import math
import dataPack

class HeadPostEstimation(object):
    """
    头部姿态识别
    """
    NOD_ACTION = 1
    SHAKE_ACTION = 2

    def __init__(self, face_detector=None):
        self.module = hub.Module(name="face_landmark_localization", face_detector_module=face_detector)
        # 头部3D关键点坐标
        self.model_points = np.array([
            [6.825897, 6.760612, 4.402142],
            [1.330353, 7.122144, 6.903745],
            [-1.330353, 7.122144, 6.903745],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],
            [1.789930, 5.393625, 4.413414],
            [-1.789930, 5.393625, 4.413414],
            [-5.311432, 5.485328, 3.987654],
            [2.005628, 1.409845, 6.165652],
            [-2.005628, 1.409845, 6.165652],
            [2.774015, -2.080775, 5.048531],
            [-2.774015, -2.080775, 5.048531],
            [0.000000, -3.116408, 6.097667],
            [0.000000, -7.415691, 4.070434],
            [-7.308957, 0.913869, 0.000000],
            [7.308957, 0.913869, 0.000000],
            [0.746313, 0.348381, 6.263227],
            [0.000000, 0.000000, 6.763430],
            [-0.746313, 0.348381, 6.263227],
        ], dtype='float')

        # 点头动作index是0， 摇头动作index是1
        # 当连续30帧上下点头动作幅度超过15度时，认为发生了点头动作
        # 当连续30帧上下点头动作幅度超过45度时，认为发生了摇头动作，由于摇头动作较为敏感，故所需幅度更大
        self._index_action = {0: 'nod', 1: 'shake'}
        self._frame_window_size = 0
        self._pose_threshold = {0: 15 / 180 * math.pi,
                                1: 45 / 180 * math.pi}
        # 头部3D投影点
        self.reprojectsrc = np.float32([
            [10.0, 10.0, 10.0],
            [10.0, 10.0, -10.0],
            [10.0, -10.0, -10.0],
            [10.0, -10.0, 10.0],
            [-10.0, 10.0, 10.0],
            [-10.0, 10.0, -10.0],
            [-10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0]])
        # 头部3D投影点连线
        self.line_pairs = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

    @property
    def frame_window_size(self):
        return self._frame_window_size

    @frame_window_size.setter
    def frame_window_size(self, value):
        assert isinstance(value, int)
        self._frame_window_size = value

    @property
    def pose_threshold(self):
        return self._pose_threshold

    @pose_threshold.setter
    def pose_threshold(self, dict_value):
        assert list(dict_value.keys()) == [0, 1, 2]
        self._pose_threshold = dict_value

    def get_face_landmark(self, image):
        """
        预测人脸的68个关键点坐标
        images(ndarray): 单张图片的像素数据
        """
        try:
            # 选择GPU运行，use_gpu=True，并且在运行整个教程代码之前设置CUDA_VISIBLE_DEVICES环境变量
            res = self.module.keypoint_detection(images=[image], use_gpu=False)
            return True, res[0]['data'][0]
        except Exception as e:
            logger.error("Get face landmark localization failed! Exception: %s " % e)
            return False, None

    def get_image_points_from_landmark(self, face_landmark):
        """
        从face_landmark_localization的检测结果抽取姿态估计需要的点坐标
        """
        image_points = np.array([
            face_landmark[17], face_landmark[21],
            face_landmark[22], face_landmark[26],
            face_landmark[36], face_landmark[39],
            face_landmark[42], face_landmark[45],
            face_landmark[31], face_landmark[35],
            face_landmark[48], face_landmark[54],
            face_landmark[57], face_landmark[8],
            face_landmark[14], face_landmark[2],
            face_landmark[32], face_landmark[33],
            face_landmark[34],
        ], dtype='float')
        return image_points

    def caculate_pose_vector(self, image_points):
        """
        获取旋转向量和平移向量
        """
        # 相机视角
        center = (self.img_size[1] / 2, self.img_size[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]],
            dtype="float")
        # 假设没有畸变
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points,
                                                                    image_points,
                                                                    camera_matrix,
                                                                    dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vector, translation_vector, camera_matrix,
                                            dist_coeffs)

        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst

    def caculate_euler_angle(self, rotation_vector, translation_vector):
        """
        将旋转向量转换为欧拉角
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
        return pitch, yaw, roll

    def classify_pose_in_euler_angles(self, datapack, poses=3):
        """
        根据欧拉角分类头部姿态(点头nod/摇头shake)
        video 表示不断产生图片的生成器
        pose=1 表示识别点头动作
        pose=2 表示识别摇头动作
        pose=3 表示识别点头和摇头动作
        """
        frames_euler = []
        self.nod_time = self.totate_time = self.shake_time = time.time()
        self.action_time = 0
        index_action = {0: [self.NOD_ACTION], 1: [self.SHAKE_ACTION]}

        # for index, img in enumerate(video(), start=1):
        self.img_size = datapack.srcFrame.shape

        success, face_landmark = self.get_face_landmark(datapack.srcFrame)

        for i, action in enumerate(index_action):
            if i == 0:
                index_action[action].append((20, int(self.img_size[0] / 2 + 110)))
            elif i == 1:
                index_action[action].append((120, int(self.img_size[0] / 2 + 110)))

        if not success:
            logger.info("Get face landmark localization failed! Please check your image!")
            return datapack

        image_points = self.get_image_points_from_landmark(face_landmark)
        success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, reprojectdst = self.caculate_pose_vector(
            image_points)

        if not success:
            logger.info("Get rotation and translation vectors failed!")
            return datapack

        # 画出投影正方体
        alpha = 0.3
        if not hasattr(self, 'before'):
            self.before = reprojectdst
        else:
            reprojectdst = alpha * self.before + (1 - alpha) * reprojectdst
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        for start, end in self.line_pairs:
            p1 = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
            p2 = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
            cv2.line(datapack.dstFrame, p1, p2, (0, 0, 255))

        # 计算头部欧拉角
        pitch, yaw, roll = self.caculate_euler_angle(rotation_vector, translation_vector)
        cv2.putText(datapack.dstFrame, "pitch: " + "{:7.2f}".format(pitch), (20, int(self.img_size[0] / 2 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(datapack.dstFrame, "yaw: " + "{:7.2f}".format(yaw), (20, int(self.img_size[0] / 2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(datapack.dstFrame, "roll: " + "{:7.2f}".format(roll), (20, int(self.img_size[0] / 2 + 70)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        '''
        for index, action in enumerate(index_action):
            cv2.putText(datapack.dstFrame, "{}".format(self._index_action[action]), index_action[action][1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), thickness=2)
        frames_euler.append([index, datapack.srcFrame, pitch, yaw, roll])

        # 转换成摄像头可显示的格式
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_rgb = srcFrame
        final_action = None
        if len(frames_euler) > self.frame_window_size:
            print("OK")
            # 比较当前头部动作欧拉角与过去的欧拉角，只有动作幅度幅度超过阈值，则判定发生相应的动作
            # picth值用来判断点头动作
            # yaw值用来判断摇头动作

            current = [pitch, yaw, roll]
            tmp = [abs(pitch), abs(yaw)]
            max_index = tmp.index(max(tmp))
            max_probability_action = index_action[max_index][0]
            for start_idx, start_img, p, y, r in frames_euler[0:int(self.frame_window_size / 2)]:
                start = [p, y, r]

                if poses & max_probability_action and abs(start[max_index] - current[max_index]) >= \
                        self.pose_threshold[max_index]:
                    print("DETECTED")
                    frames_euler = []
                    final_action = max_index
                    self.action_time = time.time()
                    # yield {self._index_action[max_index]: [(star  t_idx, start_img), (index, img)]}, img_rgb
                    # yield img_rgb
                    break
            else:
                # 丢弃过时的视频帧
                frames_euler.pop(0)
        # 动作判定发生则高亮显示0.5s
        if self.action_time != 0 and time.time() - self.action_time < 10:
            cv2.putText(datapack.dstFrame, "{}".format(self._index_action[max_index]), index_action[max_index][1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), thickness=2)

        # cv2.imshow("TEST", datapack.srcFrame)
        '''
        if pitch >= 0.35:
            datapack.isNodding = True
            cv2.putText(datapack.dstFrame, "HEAD DOWN DETECTED", (5, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        return datapack




class MyFaceDetector(object):
    """
    自定义人脸检测器
    基于PaddleHub人脸检测模型ultra_light_fast_generic_face_detector_1mb_640，加强稳定人脸检测框
    """

    def __init__(self):
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
        self.alpha = 0.75
        self.start_flag = 1

    def face_detection(self, images, use_gpu=False, visualization=False):
        # 使用GPU运行，use_gpu=True，并且在运行整个教程代码之前设置CUDA_VISIBLE_DEVICES环境变量
        result = self.module.face_detection(images=images, use_gpu=use_gpu, visualization=visualization)
        if not result[0]['data']:
            return result

        face = result[0]['data'][0]
        if self.start_flag == 1:

            self.left_s = result[0]['data'][0]['left']
            self.right_s = result[0]['data'][0]['right']
            self.top_s = result[0]['data'][0]['top']
            self.bottom_s = result[0]['data'][0]['bottom']

            self.start_flag = 0
        else:
            # 加权平均上一帧和当前帧人脸检测框位置，以稳定人脸检测框
            self.left_s = self.alpha * self.left_s + (1 - self.alpha) * face['left']
            self.right_s = self.alpha * self.right_s + (1 - self.alpha) * face['right']
            self.top_s = self.alpha * self.top_s + (1 - self.alpha) * face['top']
            self.bottom_s = self.alpha * self.bottom_s + (1 - self.alpha) * face['bottom']

        result[0]['data'][0]['left'] = self.left_s
        result[0]['data'][0]['right'] = self.right_s
        result[0]['data'][0]['top'] = self.top_s
        result[0]['data'][0]['bottom'] = self.bottom_s

        return result