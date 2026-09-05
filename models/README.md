# Model weights

The trained weights are **not stored in this git repository** because they exceed
GitHub's 100 MB file limit (and large binaries bloat clones). They are published as
a **[GitHub Release](https://github.com/liewzewei/ZYLCH_AI/releases/tag/v1.0)** instead.

## Download

**`final_statedict.pt`** (~100 MB) — the final model weights (PyTorch `state_dict`):

**→ https://github.com/liewzewei/ZYLCH_AI/releases/download/v1.0/final_statedict.pt**

Download it into this `models/` folder, e.g.:

```bash
# curl
curl -L -o models/final_statedict.pt \
  https://github.com/liewzewei/ZYLCH_AI/releases/download/v1.0/final_statedict.pt

# or wget
wget -O models/final_statedict.pt \
  https://github.com/liewzewei/ZYLCH_AI/releases/download/v1.0/final_statedict.pt
```

## How to load

```python
import torch
from src.model import CustomResNet

model = CustomResNet()
model.load_state_dict(torch.load("models/final_statedict.pt", map_location="cpu"))
model.eval()
```

Or run inference directly:

```bash
python src/infer.py data/sample/ONDE_ONDE/*.jpg --weights models/final_statedict.pt
```

## Note

The weights are hosted on the v1.0 GitHub Release above. If you retrain the model
(`notebooks/4_train.ipynb`), it saves a fresh `models/final_statedict.pt` locally —
that file is git-ignored, so it won't be committed.
