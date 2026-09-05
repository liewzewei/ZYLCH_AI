"""
Run the trained kuih classifier on one or more images.

Usage:
    python src/infer.py path/to/image.jpg
    python src/infer.py data/sample/ONDE_ONDE/*.jpg --weights models/final_statedict.pt

The weights file (`final_statedict.pt`, ~100 MB) is not committed to git — see
`models/README.md` for how to obtain it.
"""

import argparse
import glob
import sys

import torch
from PIL import Image
from torchvision.transforms import transforms

# Allow running both as `python src/infer.py` and `python -m src.infer`
try:
    from .model import CustomResNet, CLASS_NAMES
except ImportError:
    from model import CustomResNet, CLASS_NAMES

# Same preprocessing as validation/test time (resize + ImageNet normalization)
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(weights_path: str, device: str):
    model = CustomResNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict(model, image_path: str, device: str):
    img = Image.open(image_path).convert("RGB")
    tensor = _TRANSFORM(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])


def main():
    parser = argparse.ArgumentParser(description="Classify kuih images.")
    parser.add_argument("images", nargs="+", help="Image file(s) or glob(s).")
    parser.add_argument("--weights", default="models/final_statedict.pt",
                        help="Path to the model state_dict.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    # Expand any globs the shell didn't (e.g. on Windows).
    paths = []
    for pattern in args.images:
        paths.extend(glob.glob(pattern) or [pattern])

    if not paths:
        sys.exit("No images found.")

    for path in paths:
        label, conf = predict(model, path, device)
        print(f"{path}\t-> {label} ({conf:.1%})")


if __name__ == "__main__":
    main()
