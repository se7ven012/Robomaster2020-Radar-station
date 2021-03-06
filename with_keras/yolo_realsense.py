# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'with_keras/'))

import math
import colorsys
import numpy as np
from timeit import default_timer as timer

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco.names',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, num_classes)
            # make sure model, anchors and classes match
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(self.colors)
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(
                self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        # print(image_data.shape)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        results = []
        for i, c in reversed(list(enumerate(out_classes))):
            # emphasis : out_boxes[i]
            top, left, bottom, right = out_boxes[i]
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # emphasis : self.class_names[c], out_scores[i]
            results.append([self.class_names[c], round(
                out_scores[i], 2), (left, top, right, bottom)])

        end = timer()
        # print("estimated times : ", end - start)
        return results

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    lidar_map = np.zeros([800, 800], np.uint8)
    for unit in range(100, 1100, 100):
        cv2.circle(lidar_map, (0, 0), unit, 127, 2)
        cv2.circle(lidar_map, (0, 0), unit-50, 63, 2)

    video_fps = 30
    video_size = (640, 480)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        draw_image = lidar_map.copy()

        # return the result of detection
        image_PIL = Image.fromarray(color_image)
        targets = yolo.detect_image(image_PIL)

        lidar_targets = []
        for target in targets:
            label, score, left, top, right, bottom = target[0], target[1], *target[2]

            diff_x, diff_y = right-left, bottom-top
            target_point_x = left + int(round(diff_x * 0.5))
            target_point_y = top + int(round(diff_y * 0.5))

            # 取目标矩阵长宽，矩阵可以取小些排除大部分噪音,可以试一下其他值（2，3，4）
            recY = math.floor(diff_y * 0.333333333)
            recX = math.floor(diff_x * 0.333333333)

            # 取目标矩阵深度
            rect_top, rect_left = top+recY, left+recX
            # rec_depth = []
            # for rect_y in range(rect_top, rect_top + recY):
            #     for rect_x in range(rect_left, rect_left + recX):
            #         rec_depth.append(depth_frame.get_distance(rect_x, rect_y))
            rec_depth = [depth_frame.get_distance(rect_x, rect_y) for rect_x in range(
                rect_left, rect_left + recX) for rect_y in range(rect_top, rect_top + recY)]

            # 过滤掉 0 的数字
            rec_depth = np.array(rec_depth, np.float32)
            rec_depth_nonzero = list(
                filter(lambda x: int(round(x*100)) != 0, rec_depth))
            # 取平均值, 除数 +1 防止除与 0 的情况出现
            mean_depth = sum(rec_depth_nonzero) / (len(rec_depth_nonzero) + 1)

            # 将原来为零的数替换成平均值
            rec_depth[np.float32(np.round(rec_depth*100)) == 0] = mean_depth
            rec_depth.resize((recX, recY))

            # 三种滤波器都试一下效果
            result = cv2.medianBlur(rec_depth, 3)
            # result = cv2.blur(rec_depth, (5, 5))
            # #result = cv2.GaussianBlur(rec_depth_image,(7,7),0)
            depth = np.mean(result)

            #depth = depth_frame.get_distance(target_point_x, target_point_y)
            angle = math.radians(target_point_x / 8)
            lidar_x = int(round(math.sin(angle) * depth * 100))
            lidar_y = int(round(math.cos(angle) * depth * 100))

            point = (lidar_y, lidar_x)
            cv2.circle(draw_image, point, 1, 255, 2)
            cv2.putText(draw_image, text=label, org=point, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=191, thickness=2)

            # print("depth : ", depth)

        result = color_image
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        cv2.imshow("draw_image", draw_image)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("depth_colormap", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    yolo.close_session()
