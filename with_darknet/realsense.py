import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'with_darknet/'))

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2

import pyrealsense2 as rs
import darknet as dn
import pdb


def main():
    dn.set_gpu(0)
    net = dn.load_net("model_data/yolov3.cfg".encode('utf-8'), "model_data/yolov3.weights".encode('utf-8'), 0)
    meta = dn.load_meta("model_data/coco.data".encode('utf-8'))

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # try:
    #     while True:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        # continue
        pass

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    image = Image.fromarray(color_image)
    r = dn.detect(net, meta, image)
    # print(r)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', depth_image)
    cv2.imshow('color_image', color_image)

    cv2.waitKey(0)
    #         if cv2.waitKey(1) == 27:
    #             break

    #     cv2.destroyAllWindows()

    # finally:

    # Stop streaming
    pipeline.stop()


if __name__ == "__main__":
    main()
