import importlib.util
import logging
import os
import subprocess
import sys
from collections.abc import Mapping

import cv2
import numpy as np
import torch.cuda
import yaml
from tap import Tap

__all__ = ["ArgsParser", "Config"]


class ConfigArguments(Tap):
    config: str = None
    opt: str = None

    use_gpu: bool = False
    use_mp: bool = False
    use_dilation: bool = False
    max_batch_size: int = 10
    image_dir: str = None

    # Model detection arguments
    det_model_dir: str = "ckpts/onnx/det"
    det_limit_side_len: float = 960
    det_limit_type: str = "max"
    det_box_type: str = "quad"
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    det_db_score_mode: str = "fast"

    # Model recognition arguments
    rec_model_dir: str = "ckpts/onnx/rec"
    rec_batch_num: int = 6
    rec_image_shape: str = "3, 48, 320"
    rec_image_inverse: bool = True
    rec_max_text_length: int = 25
    rec_drop_score: float = 0.5

    output: str = "./inference_results"
    save_crop_res: bool = False
    crop_res_save_dir: str = "./output"

    num_processes: int = 1
    process_id: int = 0

    show_log: bool = True

    def parse_args(self, argv=None):
        args = super(ConfigArguments, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=", 1)
            if "." not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split(".")
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

        self.__dict__ = self

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


def _merge_dict(config, merge_dct):
    """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        config: dict onto which the merge is executed
        merge_dct: dct merged into config

    Returns: dct
    """
    for key, value in merge_dct.items():
        sub_keys = key.split(".")
        key = sub_keys[0]
        if key in config and len(sub_keys) > 1:
            _merge_dict(config[key], {".".join(sub_keys[1:]): value})
        elif key in config and isinstance(config[key], dict) and isinstance(value, Mapping):
            _merge_dict(config[key], value)
        else:
            config[key] = value
    return config


def print_dict(cfg, print_func=print, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, print_func, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print_func("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, print_func, delimiter + 4)
        else:
            print_func("{}{} : {}".format(delimiter * " ", k, v))


class Config(object):
    def __init__(self, config_path, BASE_KEY="_BASE_"):
        self.BASE_KEY = BASE_KEY
        self.cfg = self._load_config_with_base(config_path)

    def _load_config_with_base(self, file_path):
        """
        Load config from file.

        Args:
            file_path (str): Path of the config file to be loaded.

        Returns: global config
        """
        _, ext = os.path.splitext(file_path)
        assert ext in [".yml", ".yaml"], "only support yaml files for now"

        with open(file_path) as f:
            file_cfg = yaml.load(f, Loader=yaml.Loader)

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        if self.BASE_KEY in file_cfg:
            all_base_cfg = AttrDict()
            base_ymls = list(file_cfg[self.BASE_KEY])
            for base_yml in base_ymls:
                if base_yml.startswith("~"):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith("/"):
                    base_yml = os.path.join(os.path.dirname(file_path), base_yml)

                with open(base_yml) as f:
                    base_cfg = self._load_config_with_base(base_yml)
                    all_base_cfg = _merge_dict(all_base_cfg, base_cfg)

            del file_cfg[self.BASE_KEY]
            file_cfg = _merge_dict(all_base_cfg, file_cfg)
        file_cfg["filename"] = os.path.splitext(os.path.split(file_path)[-1])[0]
        return file_cfg

    def merge_dict(self, args):
        self.cfg = _merge_dict(self.cfg, args)

    def print_cfg(self, print_func=print):
        """
        Recursively visualize a dict and
        indenting acrrording by the relationship of keys.
        """
        print_func("----------- Config -----------")
        print_dict(self.cfg, print_func)
        print_func("---------------------------------------------")

    def save(self, p, cfg=None):
        if cfg is None:
            cfg = self.cfg
        with open(p, "w") as f:
            yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)


def _check_image_file(path):
    img_end = {"jpg", "bmp", "png", "jpeg", "rgb", "tif", "tiff", "gif", "pdf"}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def check_and_read(img_path):
    if os.path.basename(img_path)[-3:].lower() == "gif":
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logger = logging.getLogger("torchocr")
            logger.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    elif os.path.basename(img_path)[-3:].lower() == "pdf":
        import fitz
        from PIL import Image

        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
            return imgs, False, True
    return None, False, False


def binarize_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img


def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img


def get_rotate_crop_image(img, points):
    """
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    """
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_minarea_rect_crop(img, points):
    bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_a, index_b, index_c, index_d = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_a = 0
        index_d = 1
    else:
        index_a = 1
        index_d = 0
    if points[3][1] > points[2][1]:
        index_b = 2
        index_c = 3
    else:
        index_b = 3
        index_c = 2

    box = [points[index_a], points[index_b], points[index_c], points[index_d]]
    crop_img = get_rotate_crop_image(img, np.array(box))
    return crop_img


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def check_install(module_name, install_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"Warnning! The {module_name} module is NOT installed")
        print(
            f"Try install {module_name} module automatically. You can also try to install manually by pip install {install_name}."
        )
        python = sys.executable
        try:
            subprocess.check_call([python, "-m", "pip", "install", install_name], stdout=subprocess.DEVNULL)
            print(f"The {module_name} module is now installed")
        except subprocess.CalledProcessError as exc:
            raise Exception(f"Install {module_name} failed, please install manually")
    else:
        print(f"{module_name} has been installed.")


def check_gpu(use_gpu):
    if use_gpu and not torch.cuda.is_available():
        use_gpu = False
    return use_gpu


def get_check_global_params(mode):
    check_params = ["use_gpu", "max_text_length", "image_shape", "image_shape", "character_type", "loss_type"]
    if mode == "train_eval":
        check_params = check_params + ["train_batch_size_per_card", "test_batch_size_per_card"]
    elif mode == "test":
        check_params = check_params + ["test_batch_size_per_card"]
    return check_params


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        """reset"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
