



import onnxruntime

import numpy as np

class ONNXProcessor(object):

    __GPU_PROVIDER = 'CUDAExecutionProvider'
    __CPU_PROVIDER = 'CPUExecutionProvider'

    def __init__(self, onnx_detector_path:str, gpu_is_on:bool):
        self.onnx_detector_path = onnx_detector_path
        self.gpu_is_on = gpu_is_on
        self.session = None

    def initialize(self):
        if self.gpu_is_on:
            provider = self.__GPU_PROVIDER
        else:
            provider = self.__CPU_PROVIDER
        self.session = onnxruntime.InferenceSession(self.onnx_detector_path, providers=[provider])

    def process(self, data:np.ndarray) -> np.ndarray:
        ort_inputs = {self.session.get_inputs()[0].name: data}
        output = self.session.run(None, ort_inputs)
        return output