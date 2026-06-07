# Custom General EAN Architecture

This project implements an experimental image classifier called the **General EAN**
(Encoder + Abstraction Field + concept Network). Data flows through a pretrained
image encoder (EfficientNet-B0) into a compact latent vector. A learnable
**Abstraction Field** and **Concept Router** then decide which specialized
"concept modules" to activate (top-k routing), combine their outputs to make a
prediction, and a small internal **world model** learns to anticipate the latent
that comes next. An **evolution controller** tracks concept usage and can adapt
the concept population over training.

We also include a simple baseline (a small CNN trained from scratch) so you can
see whether the fancy parts actually help.

## Task

- **Task type:** image classification
- **Dataset format:** image folder (`ImageFolder`-style, one subfolder per class)
- **Recommended model:** EfficientNet-B0 (transfer learning)
- **Baseline model:** small CNN trained from scratch
- **Metrics:** accuracy, macro F1
- **Export formats:** PyTorch (`.pt`), ONNX (`.onnx`)
- **Epochs:** 20

## Dataset

By default the project downloads a public dataset automatically
(`CIFAR-10` via torchvision) and lays it out as an image folder if you do not
provide your own data. To use your own data, point `data.root` in
`configs/default.yaml` at a directory structured like:

```
my_data/
  train/
    classA/ img1.jpg ...
    classB/ img2.jpg ...
  val/
    classA/ ...
    classB/ ...
```

See `data/README.md` for details.

## Quick start

```bash
pip install -r requirements.txt

# Train the General EAN model
bash scripts/run_train.sh

# Evaluate the trained checkpoint
python -m src.evaluate --config configs/default.yaml --checkpoint runs/best.pt

# Export to PyTorch + ONNX
python -m src.export --config configs/default.yaml --checkpoint runs/best.pt
```

To train the baseline instead, set `model.name: baseline_cnn` in the config (or
pass `--override model.name=baseline_cnn`).

## Training output format

The training loop prints exactly one line per epoch:

```
epoch {n}/{total} loss={loss:.4f} val_acc={acc:.4f}
```

## Docker

```bash
docker build -t general-ean .
docker run --rm -it general-ean bash scripts/run_train.sh
```
