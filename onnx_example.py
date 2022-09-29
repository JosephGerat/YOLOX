from onnx.onnx_processor import ONNXProcessor
from onnx.preprocess_postprocess import COCOYOLOXPreprocessPostprocess
import cv2

def run(onnx_model_path, image_path):

    try:
        onnx_processor = ONNXProcessor(onnx_model_path, gpu_is_on=True)
        onnx_processor.initialize()
        preprocess_postprocess = COCOYOLOXPreprocessPostprocess((640, 640), detection_thresh=0.1)
        image = cv2.imread(image_path)
        prep = preprocess_postprocess.preprocess(image)
        out = onnx_processor.process(prep)
        output_postprocess = preprocess_postprocess.postprocess(out[0][0])
        print('DONE!')
    except Exception as e:
        print(e.args)


run("path_to_.onnx", "assets/dog.jpg")

