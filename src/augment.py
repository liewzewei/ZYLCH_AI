"""
Offline data augmentation + train/val/test split.

Takes a folder of class-subfolders of images and, for every image, writes 11
variants (the original plus 10 transforms) to `<input>_Output/`, then splits that
into train/val/test (80/10/10) at `<input>_Split_Output/`.

Usage:
    python src/augment.py "data/raw/kuih" --ratio 0.8 0.1 0.1

This is a cleaned version of the original competition script: the hard-coded input
path has been replaced with a command-line argument. Requires HEIC support for
iPhone photos (`pip install pillow-heif`).
"""

import argparse
import os
import random

import cv2 as cv
import numpy as np
import splitfolders
from PIL import Image

try:
    import pillow_heif  # noqa: F401  (registers HEIC support)
    _HEIF = True
except ImportError:
    _HEIF = False


# ── individual transforms ────────────────────────────────────────────────────

def translate(img, x, y):
    trans = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(img, trans, (img.shape[1], img.shape[0]))


def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    m = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv.warpAffine(img, m, (w, h))


def flip(img, flip_code):
    return cv.flip(img, flip_code)


def get_random_crop(img):
    """Crop a random 60% region and re-centre it on a black canvas (no interpolation)."""
    max_x = round(img.shape[1] * 0.4)
    max_y = round(img.shape[0] * 0.4)
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = img[y:y + round(img.shape[0] * 0.6), x:x + round(img.shape[1] * 0.6), :]

    canvas = np.zeros(img.shape, dtype=np.int64)
    y0 = round((img.shape[0] - crop.shape[0]) / 2)
    x0 = round((img.shape[1] - crop.shape[1]) / 2)
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1], :] = crop
    return canvas


def blur(img):
    return cv.GaussianBlur(img, (9, 9), cv.BORDER_DEFAULT)


def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    return cv.add(img, noise)


def add_salt_and_pepper_noise(img, noise_ratio=0.02):
    out = img.copy()
    h, w, _ = out.shape
    for _ in range(int(h * w * noise_ratio)):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        out[row, col] = [0, 0, 0] if np.random.rand() < 0.5 else [255, 255, 255]
    return out


def inc_brightness(img):
    return cv.addWeighted(img, 2.3, np.zeros(img.shape, img.dtype), 0, 10)


def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv.filter2D(img, -1, kernel)


def color_enhance(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[:, :, 0] = hsv[:, :, 0] * 0.7
    hsv[:, :, 1] = hsv[:, :, 1] * 1.5
    hsv[:, :, 2] = hsv[:, :, 2] * 0.5
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


# ── pipeline ──────────────────────────────────────────────────────────────────

def _read(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".heic":
        if not _HEIF:
            raise RuntimeError("HEIC image found but pillow-heif is not installed.")
        heif = pillow_heif.read_heif(path)
        pil = Image.frombytes(heif.mode, heif.size, heif.data, "raw",
                              heif.mode, heif.stride)
        return cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
    return cv.imread(path)


def augment_image(path):
    img = cv.resize(_read(path), (224, 224))
    return [
        img,
        translate(img, np.random.randint(-75, 75), np.random.randint(-75, 75)),
        rotation(img, np.random.randint(-90, 90)),
        flip(img, np.random.randint(0, 2)),
        get_random_crop(img),
        blur(img),
        add_gaussian_noise(img),
        add_salt_and_pepper_noise(img),
        inc_brightness(img),
        sharpen(img),
        color_enhance(img),
    ]


def image_augment(input_folder, split_ratio, seed=1427):
    parent = os.path.dirname(os.path.abspath(input_folder))
    out_folder = os.path.join(parent, f"{os.path.basename(input_folder)}_Output")
    os.makedirs(out_folder, exist_ok=True)

    for class_no, image_class in enumerate(os.listdir(input_folder)):
        class_out = os.path.join(out_folder, image_class)
        os.makedirs(class_out, exist_ok=True)
        class_in = os.path.join(input_folder, image_class)
        for image_no, image in enumerate(os.listdir(class_in)):
            for aug_no, aug in enumerate(augment_image(os.path.join(class_in, image))):
                cv.imwrite(os.path.join(class_out, f"{class_no}_{image_no}_{aug_no}.jpg"), aug)

    split_out = os.path.join(parent, f"{os.path.basename(input_folder)}_Split_Output")
    os.makedirs(split_out, exist_ok=True)
    splitfolders.ratio(out_folder, output=split_out, seed=seed,
                       ratio=tuple(split_ratio), group_prefix=None)
    print(f"Done. Augmented -> {out_folder}\n      Split     -> {split_out}")


def main():
    p = argparse.ArgumentParser(description="Augment images ×11 and split train/val/test.")
    p.add_argument("input_folder", help="Folder containing one subfolder per class.")
    p.add_argument("--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=1427)
    args = p.parse_args()
    image_augment(args.input_folder, args.ratio, args.seed)


if __name__ == "__main__":
    main()
