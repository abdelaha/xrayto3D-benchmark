# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""
script to train various architectures on given dataset
(defined by training and validation filepaths).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import monai.data.meta_obj as monai_meta_obj
import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import list_data_collate
from monai.utils.misc import set_determinism
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

import wandb
import XrayTo3DShape
from XrayTo3DShape import (
    AutoencoderExperiment,
    BaseExperiment,
    CustomAutoEncoder,
    TLPredictorExperiment,
    anatomy_resolution_dict,
    get_anatomy_from_path,
    get_dataset,
    get_loss,
    get_model,
    get_model_config,
    get_transform_from_model_name,
    model_experiment_dict,
    printarr,
)

np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


def parse_training_arguments():
    """reads various options from commandline

    Returns:
        dict: (key,value) pairs of option and corresponding value
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("trainpaths")
    parser.add_argument("valpaths")
    parser.add_argument("--testpaths", default=None)
    parser.add_argument("--model_name")
    parser.add_argument("--loss")
    parser.add_argument("--size", type=int)
    parser.add_argument("--res", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)

    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--steps", default=5000, type=int)

    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--logger", choices=["csv", "wandb"], default="wandb")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Checkpoint root; each logger version gets its own subdirectory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wandb-project", default="2d-3d-benchmark")

    parser.add_argument("--lambda_bce", default=1.0)
    parser.add_argument("--lambda_dice", default=1.0)
    parser.add_argument("--make_sparse", default=False, action="store_true")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--dropout", default=False, action="store_true")
    parser.add_argument("--load_autoencoder_from", default="", type=str)
    parser.add_argument('--load_model_from', default=None,type=str)
    parser.add_argument("--top_k_checkpoints", default=3, type=int)

    parser.add_argument("--precision", default=32, type=int)

    args = parser.parse_args()

    if args.precision == 16:
        args.precision = "bf16"
    return args


def update_args(args):
    """update sensible values for remaining arguments using
    the argument values provided. Perform sanity check of arguments.

    Args:
        args (dict): (option,value) pair
    """
    args.anatomy = get_anatomy_from_path(args.trainpaths)

    # add dropout to tag if exists
    # if args.dropout:
    #     args.tags.append('dropout')
    if 'dropout' in args.tags:
        args.dropout = True
    # # assert the resolution and size agree for each anatomy
    # orig_size, orig_res = anatomy_resolution_dict[args.anatomy]
    # assert int(args.size * args.res) == int(
    #     orig_size * orig_res
    # ), f"({args.size},{args.res}) does not match ({orig_size},{orig_res})"
    args.experiment_name = model_experiment_dict[args.model_name]

    if args.accelerator == "cuda":
        args.accelerator = "gpu"
    if args.accelerator in ("cpu", "mps") and args.precision != 32:
        raise ValueError("Use --precision 32 for CPU/MPS runs")


if __name__ == "__main__":
    args = parse_training_arguments()
    update_args(args)
    torch.set_num_threads(args.threads)

    # print commandline arguments: all print outputs are logged by wandb
    print(sys.argv)
    print(args)

    SEED = 12345
    lr = args.lr
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    ANATOMY = args.anatomy
    LOSS_NAME = args.loss
    IMG_RESOLUTION = args.res
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = args.wandb_project
    model_name = args.model_name
    experiment_name = args.experiment_name
    WANDB_EXPERIMENT_GROUP = args.model_name
    WANDB_TAGS = [WANDB_EXPERIMENT_GROUP, ANATOMY, LOSS_NAME, *args.tags]

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)

    train_transforms = get_transform_from_model_name(
        model_name, image_size=IMG_SIZE, resolution=IMG_RESOLUTION
    )

    train_loader = DataLoader(
        get_dataset(args.trainpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        get_dataset(args.valpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=list_data_collate,
    )

    model = get_model(
        model_name=args.model_name, image_size=IMG_SIZE, dropout=args.dropout
    )
    loss_function = get_loss(
        loss_name=LOSS_NAME,
        anatomy=ANATOMY,
        image_size=IMG_SIZE,
        lambda_bce=args.lambda_bce,
        lambda_dice=args.lambda_dice,
        device=f"cuda:{args.gpu}" if args.accelerator == "gpu" else args.accelerator,
    )
    optimizer = Adam(model.parameters(), lr)

    # load pytorch lightning module
    experiment: BaseExperiment = getattr(XrayTo3DShape.experiments, experiment_name)(
        model, optimizer, loss_function, BATCH_SIZE
    )
    if experiment_name == CustomAutoEncoder.__name__:
        experiment.make_sparse = args.make_sparse
    if experiment_name == TLPredictorExperiment.__name__:
        ae_model = get_model(model_name=CustomAutoEncoder.__name__, image_size=IMG_SIZE)
        if Path(args.load_autoencoder_from).exists():
            checkpoint = torch.load(args.load_autoencoder_from, map_location="cpu", weights_only=False)
        else:
            raise ValueError(
                f"autoencoder checkpoint {args.load_autoencoder_from} does not exist"
            )
        for key in list(checkpoint["state_dict"].keys()):
            # model.layer1.conv1 -> layer1.conv1
            modified_key = key.replace("model.", "")
            value = checkpoint["state_dict"].pop(key)
            checkpoint["state_dict"][modified_key] = value
        if "loss_function.pos_weight" in checkpoint["state_dict"]:
            checkpoint["state_dict"].pop("loss_function.pos_weight")
        ae_model.load_state_dict(checkpoint["state_dict"])
        experiment.set_decoder(ae_model)  # type: ignore

    if args.load_model_from:
        if Path(args.load_model_from).exists():
            checkpoint = torch.load(args.load_model_from, map_location="cpu", weights_only=False)
            # Transfer model weights, preserving the new loss and fresh optimizer.
            state = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items()
                     if k.startswith("model.")}
            model.load_state_dict(state, strict=True)
            print(f'loaded model checkpoint from {args.load_model_from}')
        else:
            raise ValueError(f'could not load model checkpoint from {args.load_model_from}')
        
    # run a sanity check
    batch = next(iter(train_loader))
    if args.experiment_name != AutoencoderExperiment.__name__:
        seg_meta_dict = experiment.get_segmentation_meta_dict(batch)
    batch_input, batch_output = experiment.get_input_output_from_batch(batch)

    pred_logits = experiment.model(*batch_input)
    if experiment_name == AutoencoderExperiment.__name__:
        pred_logits, latent_vec = pred_logits
    if experiment_name == TLPredictorExperiment.__name__:
        pred_logits = ae_model.latent_vec_decode(pred_logits)  # type: ignore
    loss = experiment.loss_function(pred_logits, batch_output).item()  # type: ignore
    input_zero = batch_input[0]
    printarr(pred_logits, batch_output, input_zero, loss)
    print(
        f"training samples {len(train_loader.dataset)} validation samples {len(val_loader.dataset)}"
    )
    print(f"Track meta data : {monai_meta_obj.get_track_meta()}")
    if args.debug:
        sys.exit()
    # loggers
    if args.logger == "wandb":
        wandb_logger = WandbLogger(
            save_dir=args.output_dir,
            project=WANDB_PROJECT,
            group=WANDB_EXPERIMENT_GROUP,
            tags=WANDB_TAGS,
        )
        wandb_logger.watch(model, log_graph=False)
    else:
        wandb_logger = CSVLogger(save_dir=args.output_dir, name="training")
    MODEL_CONFIG = get_model_config(model_name, IMG_SIZE)
    # save hyperparameters
    HYPERPARAMS = {
        "IMG_SIZE": IMG_SIZE,
        "RESOLUTION": IMG_RESOLUTION,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": lr,
        "SEED": SEED,
        "ANATOMY": ANATOMY,
        "MODEL_NAME": model_name,
        "LOSS": LOSS_NAME,
        "EXPERIMENT_NAME": experiment_name,
    }
    HYPERPARAMS.update(MODEL_CONFIG)
    experiment.save_hyperparameters(HYPERPARAMS)
    wandb_logger.log_hyperparams(HYPERPARAMS)
    CHECKPOINT_FILENAME = (
        "epoch={epoch}-step={step}-val_loss={val/loss:.2f}"
        if experiment_name == CustomAutoEncoder.__name__
        else "epoch={epoch}-step={step}-val_dice={val/dice:.2f}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(args.checkpoint_dir) / f"version_{wandb_logger.version}")
        if args.checkpoint_dir else None,
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=args.top_k_checkpoints,
        filename=CHECKPOINT_FILENAME,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        max_epochs=NUM_EPOCHS,
        devices=[args.gpu] if args.accelerator == "gpu" else 1,
        deterministic=False,
        log_every_n_steps=1,
        logger=[wandb_logger],
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_checkpointing=True,
        max_steps=args.steps,
    )

    trainer.fit(experiment, train_loader, val_loader)

    if args.testpaths:
        import nibabel as nib
        test_loader = DataLoader(
            get_dataset(args.testpaths, transforms=train_transforms),
            batch_size=BATCH_SIZE, num_workers=args.num_workers, shuffle=False,
            collate_fn=list_data_collate,
        )
        # Reload the best saved weights to verify the trained-checkpoint path too.
        best = checkpoint_callback.best_model_path
        metrics = trainer.validate(experiment, test_loader, ckpt_path=best)
        results = trainer.predict(experiment, test_loader, ckpt_path=best)
        predictions_dir = Path(wandb_logger.log_dir) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        index = 0
        for result in results:
            for prediction, affine in zip(result["pred"], result["target_affine"]):
                array = prediction[0].detach().cpu().numpy().astype(np.uint8)
                nib.save(nib.Nifti1Image(array, affine.detach().cpu().numpy()),
                         predictions_dir / f"row_{index:04d}_pred.nii.gz")
                index += 1
        (predictions_dir.parent / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Saved {index} predictions to {predictions_dir}")

    if args.logger == "wandb":
        wandb.finish()
