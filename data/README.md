# Data

This project expects an **image folder** layout (one subdirectory per class).

## Option A: Automatic download (default)

If `data.auto_download: true` and `data.root` is empty, the code downloads
**CIFAR-10** via torchvision and writes it out as image folders:

```
data/dataset/
  train/
    airplane/  ... .png
    automobile/ ...
    ...
  val/
    airplane/ ...
    ...
```

## Option B: Bring your own data

Place your images under `data.root` using this structure:

```
<root>/
  train/
    classA/ img1.jpg img2.jpg ...
    classB/ ...
  val/
    classA/ ...
    classB/ ...
```

If only a `train/` folder exists (no `val/`), the loader automatically carves a
validation split out of training data using `data.val_split`.

Supported image extensions: `.jpg .jpeg .png .bmp .gif .webp`.
