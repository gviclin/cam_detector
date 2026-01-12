#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

import cv2
import numpy as np
import openvino as ov

CONF_THRESHOLD = 0.20


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # # Parsing and validation of input arguments
    # if len(sys.argv) != 4:
    #     log.info(f'Usage: {sys.argv[0]} <path_to_model> <path_to_image> <device_name>')
    #     return 1

    # model_path = sys.argv[1]
    # image_path = sys.argv[2]
    # device_name = sys.argv[3]

    model_path = "yolov8n.onnx"
    image_path = "person.jpg"
    device_name = "CPU"


# --------------------------- Step 1. Initialize OpenVINO Runtime Core ------------------------------------------------
    log.info('Creating OpenVINO Runtime Core')
    core = ov.Core()

# --------------------------- Step 2. Read a model --------------------------------------------------------------------
    log.info(f'Reading the model: {model_path}')
    # (.xml and .bin files) or (.onnx file)
    model = core.read_model(model_path)

    if len(model.inputs) != 1:
        log.error('Sample supports only single input topologies')
        return -1

    if len(model.outputs) != 1:
        log.error('Sample supports only single output topologies')
        return -1

# --------------------------- Step 3. Set up input --------------------------------------------------------------------
    # Read input image
    image = cv2.imread(image_path)
    print("Original shape (BGR):", image.shape)  # HWC, BGR

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    # image = image.astype(np.float32) / 255.0

    # Add N dimension
    input_tensor = np.expand_dims(image, 0)

    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640, 640))
    # img = img.astype(np.float32) / 255.0
    # img = np.transpose(img, (2,0,1))  # HWC -> CHW
    # input_tensor = np.expand_dims(img, 0)       # batch dimension

    print("input_tensor shape :", input_tensor.shape)  # HWC, BGR


# --------------------------- Step 4. Apply preprocessing -------------------------------------------------------------
    ppp = ov.preprocess.PrePostProcessor(model)

    _, h, w, _ = input_tensor.shape

    # 1) Set input tensor information:
    # - input() provides information about a single model input
    # - reuse precision and shape from already available `input_tensor`
    # - layout of data is 'NHWC'
    ppp.input().tensor() \
        .set_shape(input_tensor.shape) \
        .set_element_type(ov.Type.u8) \
        .set_layout(ov.Layout('NHWC'))  # noqa: ECE001, N400

    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Here we suppose model has 'NCHW' layout for input
    ppp.input().model().set_layout(ov.Layout('NCHW'))

    # 4) Set output tensor information:
    # - precision of tensor is supposed to be 'f32'
    ppp.output().tensor().set_element_type(ov.Type.f32)

    # 5) Apply preprocessing modifying the original 'model'
    model = ppp.build()

# --------------------------- Step 5. Loading model to the device -----------------------------------------------------
    log.info('Loading the model to the plugin')
    compiled_model = core.compile_model(model, device_name)

# --------------------------- Step 6. Create infer request and do inference synchronously -----------------------------
    log.info('Starting inference in synchronous mode')
    results = compiled_model.infer_new_request({0: input_tensor})

# --------------------------- Step 7. Process output ------------------------------------------------------------------
    predictions = results[0][0]  # batch 0

    predictions = np.transpose(predictions, (1, 0))

    # Pour chaque detection
    for idx, det in enumerate(predictions): 
        # print(f"idx {idx}")
        x, y, w, h = det[0:4]
        obj_conf = float(det[4])
        class_scores = det[5:] * obj_conf  # objectness × class probabilities
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        # if (obj_conf > 0.0001):
        #     print(f"obj_conf {obj_conf}")
    
        if confidence < 0.01:
            continue        

        print(f"idx {idx}, class_id={class_id}, obj_conf {obj_conf:.3f}, confidence={confidence:.3f}, bbox={x,y,w,h}")




# ----------------------------------------------------------------------------------------------------------------------
    log.info('This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())