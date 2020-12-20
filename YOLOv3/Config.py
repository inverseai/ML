IMAGE_DIR = "/content/YOLOv3/dataset/VOCdevkit/VOC2012/JPEGImages/"
ANNOT_DIR = "/content/YOLOv3/dataset/VOCdevkit/VOC2012/Annotations/"
TEST_DIR  = "/content/YOLOv3/dataset/VOCdevkit/VOC2012/JPEGImages/"
WEIGHT_FILE="/content/YOLOv3/weights/yolov3.weights"

CLASSES=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
        'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush']
NUM_CLASSES=85

ANCHORS=[[[10,  13], [16,   30], [33,   23]],
              [[30,  61], [62,   45], [59,  119]],
              [[116, 90], [156, 198], [373, 326]]]

STRIDES=[8, 16, 32]
INPUT_SIZE=416
TRAIN_BATCH_SIZE=4
TEST_BATCH_SIZE=1
ANCHOR_PER_SCALE=3
MAX_BBOX_PER_SCALE=100
IOU_LOSS_THRESH=0.45
SCORE_THRESHOLD=0.40
TRAIN_WARMUP_EPOCHS=2
TRAIN_EPOCHS=100
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6