import sys
import os

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2

import pyrealsense2 as rs
import darknet as dn
import pdb


def main():
    dn.set_gpu(0)
    net = dn.load_net("cfg/yolov3.cfg".encode('utf-8'), "cfg/yolov3.weights".encode('utf-8'), 0)
    meta = dn.load_meta("cfg/coco.data".encode('utf-8'))

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

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
        depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    # image = Image.fromarray(color_image)
    # r = dn.detect(net, meta, image)
    # print(r)

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images)
    cv2.imshow('color_image', color_image)
    # cv2.imshow('depth_image', depth_image)

    cv2.waitKey(0)
    #         if cv2.waitKey(1) == 27:
    #             break

    #     cv2.destroyAllWindows()

    # finally:

    # Stop streaming
    pipeline.stop()


if __name__ == "__main__":
    main()
