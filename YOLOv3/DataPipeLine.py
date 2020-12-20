import os
import cv2
import random
import copy
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as xTree
from Utils import image_preprocess
from Config import*

class TrainBatchGenerator(object):
    def __init__(self, image_dir=IMAGE_DIR, annot_dir=ANNOT_DIR):
        self.annot_dir = annot_dir
        self.image_dir = image_dir
        self.input_size = INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE
        self.strides = np.array(STRIDES)
        self.classes=CLASSES
        self.num_classes = len(self.classes)
        self.anchors = (np.array(ANCHORS).T/self.strides).T
        self.anchor_per_scale = ANCHOR_PER_SCALE
        self.max_bbox_per_scale = MAX_BBOX_PER_SCALE
        self.annotations = self.parse_annotation()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        print(self.num_batchs)

    def parse_annotation(self):
        train_data=[]
        for annot_name in sorted(os.listdir(self.annot_dir)):
            split=annot_name.split('.')
            img_name=split[0]

            img_path=self.image_dir+img_name
            new_data={ 'object' : [] }
            if os.path.exists(img_path+'.jpg'):
                img_path=img_path+'.jpg'
            elif os.path.exists(img_path+'.JPG'):
                img_path=img_path+'.JPG'
            elif os.path.exists(img_path+'.jpeg'):
                img_path=img_path+'.jpeg'
            elif os.path.exists(img_path+'.png'):
                img_path=img_path+'.png'
            elif os.path.exists(img_path+'.PNG'):
                img_path=img_path+'.PNG'
            else:
                print('image path not exis')
                assert(False)
            new_data['image_path']=img_path
            annot=xTree.parse(self.annot_dir+annot_name)
            for elem in annot.iter():
                if elem.tag == 'width':
                    new_data['width']=int(elem.text)
                if elem.tag=='height':
                    new_data['height']=int(elem.text)
                if elem.tag=='object':
                    obj={}
                    for attr in list(elem):
                        if attr.tag=='name':
                            obj['name']=attr.text
                        if attr.tag=='bndbox':
                            for dim in list(attr):
                                obj[dim.tag]=int(round(float(dim.text)))
                    new_data['object'].append(obj)
            train_data.append(new_data)
        return train_data
    
    def __iter__(self):
        return self
    def __len__(self):
        return self.num_batchs

    def __next__(self):
        with tf.device('/cpu:0'):
            self.output_size= self.input_size // self.strides
            batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3), dtype=np.float32)
            batch_label_sbbox = np.zeros((self.batch_size, self.output_size[0], self.output_size[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.output_size[1], self.output_size[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.output_size[2], self.output_size[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_data(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))
            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]
            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return image, bboxes

    def parse_data(self, train_instance):
        image_name = train_instance['image_path']
        image = cv2.imread(image_name)
        all_objs = copy.deepcopy(train_instance['object'])
        boxes = []
        for i, obj in enumerate(all_objs):
            l=int(self.classes.index(obj['name']))
            x_min=float(obj['xmin'])
            x_max=float(obj['xmax'])
            y_min=float(obj['ymin'])
            y_max=float(obj['ymax'])
            boxes.append([x_min, y_min, x_max, y_max, l])
        boxes = np.asarray(boxes, dtype=np.int32)
        
        #image =self.random_color_distort(image)
        image, boxes = self.random_horizontal_flip(np.copy(image), np.copy(boxes))
        image, boxes = self.random_crop(np.copy(image), np.copy(boxes))
        image, boxes = self.random_translate(np.copy(image), np.copy(boxes))
        image, boxes =image_preprocess(np.copy(image), [self.input_size, self.input_size], np.copy(boxes))
        return image, boxes

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.output_size[i], self.output_size[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area * 1.0 / tf.maximum(union_area, 1e-24)

class TestBatchGenerator(object):
    def __init__(self, test_dir=TEST_DIR):
        self.test_dir = test_dir
        self.input_size = INPUT_SIZE
        self.images=os.listdir(self.test_dir)
        self.num_samples = len(self.images)
        self.index=0
        print(self.num_samples)
    
    def __iter__(self):
        return self
    def __len__(self):
        return self.num_batchs
        
    def __next__(self):
        with tf.device('/cpu:0'):
            if self.index>=self.num_samples:
                raise StopIteration
                self.index=0
            test_path=self.test_dir+self.images[self.index]
            original_image = cv2.imread(test_path)
            image = image_preprocess(np.copy(original_image), [self.input_size, self.input_size])
            image=image[np.newaxis, :, :, :]
            self.index+=1
            return original_image, image
