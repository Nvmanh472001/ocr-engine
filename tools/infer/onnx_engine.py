import os

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class ONNXEngine:
    provider = ort.get_available_providers()

    def __init__(self, onnx_path, use_gpu=False) -> None:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        self.inference_session = ort.InferenceSession(
            onnx_path,
            providers=self.provider if use_gpu else ["CPUExecutionProvider"],
        )

        self.input_name = [node.name for node in self.inference_session.get_inputs()]
        self.output_name = [node.name for node in self.inference_session.get_outputs()]

    def run(self, image_np: np.ndarray) -> NDArray[np.float32]:
        input_feed = {name: image_np for name in self.input_name}
        return self.inference_session.run(self.output_name, input_feed=input_feed)
