import os
from typing import Dict

import onnx
import tensorflow as tf
import torch
from ofa.imagenet_classification.elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from onnx_tf.backend import prepare


def arch2tflite(arch: Dict, name: str, out_dir: str):
    onnx_model_path = os.path.join(out_dir, name + ".onnx")
    tf_model_path = os.path.join(out_dir, name + ".pb")
    tflite_model_path = os.path.join(out_dir, name + ".tflite")

    # model decode
    mobilenetv3 = OFAMobileNetV3(
        ks_list=arch["ks"],
        expand_ratio_list=arch["e"],
        depth_list=arch["d"],
    )
    image_size = arch["r"]

    # model to onnx
    dummy_input = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(mobilenetv3, dummy_input, onnx_model_path, input_names=["input"])

    # model to tf
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)

    # model to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "exported_models/mobilenetv3_1.pb"
    )
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    archs = torch.load("data/ofa/ofa_archs.pt")["arch"]

    for i in len(archs):
        arch2tflite(archs[i], f"mobilenetv3_{i}", "exported_models")
