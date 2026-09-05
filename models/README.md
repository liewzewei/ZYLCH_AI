# Model weights

The trained weights are **not stored in this git repository** because they are
~100 MB each (GitHub blocks files over 100 MB, and large binaries bloat clones).

## Files

| File | Size | Description |
|---|---|---|
| `final_statedict.pt` | ~100 MB | **Recommended.** Final model weights (`state_dict`). Load into `CustomResNet`. |
| `final_entire_model.pt` | ~100 MB | Whole pickled model object (needs the exact class + library versions to load). |

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

## How to distribute the weights (pick one)

- **GitHub Releases (recommended):** attach `final_statedict.pt` to a release and
  link it here. Release assets can be up to 2 GB and don't bloat the repo.
- **Git LFS:** `git lfs track "*.pt"` if you want the weights versioned in-repo.
- **Google Drive:** host the file and download it with `gdown` (this project's
  Colab notebook already uses that approach).

<!-- TODO: paste the download link here once the weights are hosted. -->
