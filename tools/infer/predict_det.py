import json
import os
import sys
import time
from typing import List, Tuple, Union

import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

import numpy as np
from numpy.typing import NDArray

from data.imaug import create_operators, transform
from postprocess import build_post_process
from tools.infer.onnx_engine import ONNXEngine
from tools.utility import (
    Config,
    ConfigArguments,
    check_and_read,
    check_gpu,
    get_image_file_list,
)
from utils.logging import get_logger
from utils.visual import draw_text_det_res

logger = get_logger()


class TextDetector(ONNXEngine):
    def __init__(self, args: ConfigArguments) -> None:
        if args.det_model_dir is None or not os.path.exists(args.det_model_dir):
            raise Exception(f"args.model_dir is set to {args.det_model_dir}, but it does not exist")

        onnx_path = os.path.join(args.det_model_dir, "prunning_model.onnx")
        super(TextDetector, self).__init__(onnx_path, args.use_gpu)

        self.args = args

        conf_path = os.path.join(args.det_model_dir, "config.yml")
        conf = Config(config_path=conf_path).cfg

        pre_process_list: List[dict] = [
            {
                "DetResizeForTest": {
                    "limit_size": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]

        self.preprocess_ops = create_operators(op_param_list=pre_process_list)
        self.postprocess_ops = build_post_process(conf["PostProcess"])

    def order_points_clockwise(self, pts: NDArray[np.float32]) -> NDArray[np.float32]:
        rect: NDArray[np.float32] = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points: NDArray[np.float32], img_height: int, img_width: int) -> NDArray[np.float32]:
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes: NDArray[np.float32], image_shape: Tuple[int, int]) -> NDArray[np.float32]:
        img_height, img_width = image_shape[0:2]
        dt_boxes_new: List[NDArray[np.float32]] = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(
        self,
        dt_boxes: NDArray[np.float32],
        image_shape: Tuple[int, int],
    ) -> NDArray[np.float32]:
        img_height, img_width = image_shape[0:2]
        dt_boxes_new: List[NDArray[np.float32]] = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img: NDArray[np.float32]) -> Union[Tuple[NDArray[np.float32], float], Tuple[None, float]]:
        ori_im: NDArray[np.float32] = img.copy()
        data: dict = {"image": img}

        st: float = time.time()

        data = transform(data, self.preprocess_ops)
        img, shape_list = data
        print(f"norm: {shape_list=}")
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        print(f"expand: {shape_list=}")
        img = img.copy()

        preds: NDArray[np.float32] = self.run(img)

        post_result: dict = self.postprocess_ops({"maps": preds[0]}, shape_list)
        dt_boxes: NDArray[np.float32] = post_result[0]["points"]

        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_im.shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        et: float = time.time()
        return dt_boxes, et - st


def main(args):
    args.use_gpu = check_gpu(args.use_gpu)

    image_file_list = get_image_file_list(args.image_dir)
    text_detector = TextDetector(args)

    total_time = 0
    save_res_path = args.output
    os.makedirs(save_res_path, exist_ok=True)

    # if args.warmup:
    #     img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
    #     for i in range(2):
    #         res = text_detector(img)

    with open(os.path.join(save_res_path, "inference_det.txt"), "w") as fout:
        for image_file in image_file_list:
            img, flag, _ = check_and_read(image_file)
            if not flag:
                img = cv2.imread(image_file)
            if img is None:
                logger.info(f"error in loading image:{image_file}")
                continue

            tic = time.time()
            dt_boxes, _ = text_detector(img)
            elapse = time.time() - tic
            total_time += elapse

            dt_boxes_json = []
            # write result
            for box in dt_boxes:
                tmp_json = {"transcription": "", "points": np.array(box).tolist()}
                dt_boxes_json.append(tmp_json)
            out_str = f"{image_file}\t{json.dumps(dt_boxes_json)}"
            fout.write(out_str + "\n")

            logger.info(out_str)
            logger.info(f"The predict time of {image_file}: {elapse}")

            src_im = draw_text_det_res(dt_boxes, image_file)
            img_name_pure = os.path.split(image_file)[-1]
            img_path = os.path.join(save_res_path, "det_res_prunning_{}".format(img_name_pure))
            cv2.imwrite(img_path, src_im)


if __name__ == "__main__":
    inference_args = ConfigArguments()
    main(inference_args.parse_args())
