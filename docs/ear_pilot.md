# Ear UNet pilot

Run from the project root (Python 3.12):

```bash
python3 -m venv .venv-train
.venv-train/bin/python -m pip install -r xrayto3D-benchmark/requirements-ear.txt
cd xrayto3D-benchmark
../.venv-train/bin/python train.py \
  ../data/ear/manifests/ear_train.csv ../data/ear/manifests/ear_val.csv \
  --testpaths ../data/ear/manifests/ear_test.csv \
  --model_name UNet --loss DiceLoss --size 64 --res 0.5 \
  --batch_size 2 --lr 0.0002 --steps 200 \
  --accelerator cpu --precision 32 --num_workers 0 --logger csv \
  --output_dir ../data/ear/pilot_runs \
  --checkpoint_dir ../data/checkpoint/ear \
  --load_model_from '/Volumes/Extreme SSD/experiment_log_dir/femur/UNet/checkpoints/epoch=38-step=3822-val_dice=0.93.ckpt'
```

Outputs: CSV metrics, hyperparameters, test metrics, and NIfTI predictions
under `data/ear/pilot_runs/training/version_*/`.
Best/last checkpoints go to `data/checkpoint/ear/version_*/`.
Predictions use the transformed target grid and its physical affine.

For AWS: use `--accelerator cuda --gpu 0`, a suitable batch size, and local
paths to the downloaded dataset/checkpoint. `--precision 16` enables bfloat16
mixed precision; use it only on a GPU supporting bfloat16. CSV logging needs
no external account; `--logger wandb` enables W&B.

Checkpoint initialization loads model weights strictly and starts a fresh
optimizer. It does not resume the source organ's training state.

The repeated patient in all splits is only for plumbing; these metrics do
not measure generalization. Partial final batches are retained.
Use `--steps 20` for a quick smoke run.
