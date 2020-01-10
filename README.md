# Robomasters 2020 radar system

- [Robomasters 2020 radar system](#robomasters-2020-radar-system)
  - [with keras-yolo3](#with-keras-yolo3)
    - [Environment](#environment)
    - [Quick Start](#quick-start)
    - [Training](#training)
  - [with_darknet](#withdarknet)
    - [Environment](#environment-1)
    - [Quick Start](#quick-start-1)
  - [Achievement](#achievement)

---

## with keras-yolo3

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).

### Environment

- Python 3.5.2
- Keras 2.1.5
- tensorflow 1.6.0
- pyrealsense2 2.29.0.1124

### Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```bash
wget https://pjreddie.com/media/files/yolov3.weights -P model_data/
python model_data/convert.py model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo.h5
python with_keras/yolo_video.py
```

### Training
1. Prepare your data.
    We suggest using [OpenLabeling](https://github.com/Cartucho/OpenLabeling) for image labeling.

2. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

3. Make sure you run `python model_data/convert.py -w model_data/yolov3.cfg model_data/yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

4. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74 -P model_data/`  
    2. rename it as darknet53.weights  
    3. `python model_data/convert.py -w model_data/darknet53.cfg model_data/darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## with_darknet

### Environment

- Python 3.5.2
- pyrealsense2 2.29.0.1124

### Quick Start

1. Run YOLO detection.

```bash
wget https://pjreddie.com/media/files/yolov3.weights -P model_data/
python with_darknet/realsense.py
```

## Achievement

So far it looks like:

<div align="center">
  <img src=imgs/scanner_overall.gif width="720px"/>
</div>

A filtering algorithm will be develope to get high precision and stable depth value.

<div align="center">
  <img src=imgs/scanner_holdon.gif width="720px"/>
</div>

We expected to use multiple cameras fixed on a supporter to achieve 360-degree real-time scanning. This can also be achieved by changing camera direction to enable scanning in case of short of cameras.

