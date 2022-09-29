
from typing import Tuple

import cv2
import numpy as np

from onnx.ipreprocess_postprocess import IPreprocessPostprocess


class COCOYOLOXPreprocessPostprocess(IPreprocessPostprocess):

    def __init__(self, input_numpy_shape:Tuple, detection_thresh=0.33):
        '''
        :param input_numpy_shape: for image like (640, 640) -> for resize/preprocess
        :param output_numpy_shape: for preprocess (8400, 85)
        '''
        self.ratio = None
        self.detection_thresh = detection_thresh
        self.input_numpy_shape = input_numpy_shape
        self.input_image_shape = None


    def __preproc(self, img, input_size, swap=(2, 0, 1)):
        # TODO REFACTOR / SPEED UP
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_NEAREST,
        )
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r


    def __nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def __multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.__nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets

    def __demo_postprocess(self, outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs


    def preprocess(self, data:np.ndarray) -> np.ndarray:
        self.input_image_shape = data.shape
        padded_img, self.ratio = self.__preproc(data, self.input_numpy_shape)
        return np.expand_dims(padded_img, axis=0)

    def postprocess(self, data:np.ndarray):

        # TODO
        predictions = self.__demo_postprocess(data, self.input_numpy_shape)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= self.ratio
        dets = self.__multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=self.detection_thresh)

        bboxes = []
        normalized_bboxes = []
        scores = []
        classes = []
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

            for i in range(final_boxes.shape[0]):

                box = final_boxes[i]
                score = final_scores[i]
                cls = final_cls_inds[i]

                if score < self.detection_thresh:
                    continue
                #if cls != 0:
                #    continue

                x0 = max(int(box[0]), 0)
                y0 = max(int(box[1]), 0)
                x1 = min(int(box[2]), self.input_image_shape[1])
                y1 = min(int(box[3]), self.input_image_shape[0])

                bboxes.append([x0, y0, x1, y1])
                normalized_bboxes.append(
                    [
                        x0/self.input_image_shape[1],
                        y0/self.input_image_shape[0],
                        x1/self.input_image_shape[1],
                        y1/self.input_image_shape[0]
                    ]
                )
                scores.append(score)
                classes.append(cls)
        #print(normalized_bboxes)
        return bboxes, normalized_bboxes, scores, classes