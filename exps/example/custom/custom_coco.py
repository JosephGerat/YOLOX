
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
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/ball_ds"
        self.train_ann = "instances_default.json"
        self.val_ann = "instances_default_valid.json"
        self.image_dir_name = "images"

        self.num_classes = 2

        self.max_epoch = 300
        self.data_num_workers = 1
        self.eval_interval = 1
