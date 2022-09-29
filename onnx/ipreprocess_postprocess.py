
import abc

import numpy as np

class IPreprocessPostprocess:

    @abc.abstractmethod
    def preprocess(self, data:np.ndarray):
        pass

    @abc.abstractmethod
    def postprocess(self, data:np.ndarray):
        pass