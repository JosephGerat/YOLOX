import time

import numpy as np

from onnx.onnx_processor import ONNXProcessor
from onnx.preprocess_postprocess import COCOYOLOXPreprocessPostprocess
import cv2

def run(onnx_model_path, image_path):
    print(f'Image : {image_path}')
    try:
        onnx_processor = ONNXProcessor(onnx_model_path, gpu_is_on=True)
        onnx_processor.initialize()
        preprocess_postprocess = COCOYOLOXPreprocessPostprocess((640, 640), detection_thresh=0.1)
        image = cv2.imread(image_path)
        prep = preprocess_postprocess.preprocess(image)
        out = onnx_processor.process(prep)

        if False:
            prep = np.zeros((3, 3, 640, 640), dtype=np.float32)
            for i in range(10):
                start = time.time()
                out = onnx_processor.process(prep)
                print(f'Duration {time.time()-start}')
        bboxes, normalized_bboxes, scores, classes = preprocess_postprocess.postprocess(out[0][0])

        for index in range(len(bboxes)):
            if scores[index] > 0.4:
                cv2.rectangle(
                    image,
                    (bboxes[index][0], bboxes[index][1]),
                    (bboxes[index][2], bboxes[index][3]),
                    (255, 0, 0))
                print(f'Scores {scores[index]}')

        #cv2.imshow('Image', image)
        #cv2.waitKey(0)

    except Exception as e:
        print(e.args)

import glob
image_paths = glob.glob("datasets/images/*.jpg")

for image_path in image_paths:
    run(
        "yolox_s_custom_depth05.onnx",
        image_path
    )

