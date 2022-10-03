
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50

        #self.depth = 0.67
        #self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/lunter"
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_valid.json"
        self.image_dir_name = "images"

        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        #self.multiscale_range = 0

        self.max_epoch = 100
        self.data_num_workers = 1
        self.eval_interval = 3
        self.enable_mixup = False
        self.mosaic_scale = (0.5, 1.3)
        self.degrees = 5.0
