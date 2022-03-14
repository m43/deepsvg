import argparse
import importlib
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset

from deepsvg import utils
from deepsvg.config import _Config
from deepsvg.difflib.tensor import SVGTensor
from deepsvg.difflib.utils import plot_points
from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVG
from deepsvg.svglib.utils import make_grid
from deepsvg.utils import Stats, TrainVars, Timer, get_str_formatted_time


def train(cfg: _Config, model_name, experiment_name="", log_dir="./logs", debug=False, resume=False, eval_only=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Parameters")
    cfg.print_params()

    print("Loading dataset")
    dataset_load_function = importlib.import_module(cfg.dataloader_module).load_dataset
    train_dataset, test_dataset = dataset_load_function(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                  num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    valid_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                  num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)

    model = cfg.make_model().to(device)
    if cfg.pretrained_path is not None:
        print(f"Loading pretrained model {cfg.pretrained_path}")
        utils.load_model(cfg.pretrained_path, model, device)

    stats = Stats(num_steps=cfg.num_steps, num_epochs=cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=cfg.stats_to_print)
    train_vars = TrainVars()
    valid_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    current_time = utils.get_str_formatted_time()
    experiment_identifier = f"{model_name}__{experiment_name}__{current_time}"

    summary_writer = SummaryWriter(
        os.path.join(log_dir, "tensorboard", "debug" if debug else "full", experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_name)
    visualization_dir = os.path.join(log_dir, "visualization", model_name, experiment_name)
    utils.ensure_dir(visualization_dir)

    cfg.set_train_vars(train_vars, train_dataloader)
    cfg.set_train_vars(valid_vars, valid_dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in cfg.make_losses()]

    if resume:
        ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, device, None, optimizers, scheduler_lrs,
                                           scheduler_warmups, stats, train_vars)

    if resume and ckpt_exists:
        print(f"Resuming model at epoch {stats.epoch + 1}")
        stats.num_steps = cfg.num_epochs * len(train_dataloader)
        stats.reset_stats_to_print()
    else:
        # Run a single forward pass on the single-device model for initialization of some modules
        single_foward_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size // cfg.num_gpus, shuffle=True,
                                              drop_last=True,
                                              num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
        data = next(iter(single_foward_dataloader))
        model_args, params_dict = [data[arg].to(device) for arg in cfg.model_args], cfg.get_params(0, 0)
        model(*model_args, params=params_dict)

    model = nn.DataParallel(model)

    if eval_only:
        # dataset = ConcatDataset([train_dataset, test_dataset])
        # valid_dataloader = DataLoader(
        #     dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        #     num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
        # evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, 0, 0, summary_writer,
        #          visualization_dir)
        evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, 0, 0, summary_writer,
                 visualization_dir)
        evaluate(cfg, model, device, loss_fns, valid_vars, train_dataloader, "train", stats, 0, 0, summary_writer,
                 visualization_dir)
        return

    if stats.epoch == 0:
        evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, 0, 0, summary_writer,
                 visualization_dir)

    epoch_range = utils.infinite_range(stats.epoch) if cfg.num_epochs is None else range(stats.epoch, cfg.num_epochs)
    for epoch in epoch_range:
        print(f"Epoch {epoch + 1}")

        for n_iter, data in enumerate(train_dataloader):
            step = n_iter + epoch * len(train_dataloader)
            timer.reset()

            if cfg.num_steps is not None and step > cfg.num_steps:
                return

            model.train()
            model_args = [data[arg].to(device) for arg in cfg.model_args]
            labels = data["label"].to(device) if "label" in data else None
            params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

            for i, (loss_fn, optimizer, scheduler_lr, scheduler_warmup, optimizer_start) in enumerate(
                    zip(loss_fns, optimizers, scheduler_lrs, scheduler_warmups, cfg.optimizer_starts), 1):
                optimizer.zero_grad()

                output = model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)

                if step >= optimizer_start:
                    loss_dict["loss"].backward()
                    if cfg.grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                    optimizer.step()
                    if scheduler_lr is not None:
                        scheduler_lr.step()
                    if scheduler_warmup is not None:
                        scheduler_warmup.step()

                stats.update_stats_to_print("train", loss_dict.keys())
                stats.update("train", step, epoch, {
                    ("lr" if i == 1 else f"lr_{i}"): optimizer.param_groups[0]['lr'],
                    **loss_dict
                })

            stats.update("train", step, epoch, {
                **weights_dict,
                "time": timer.get_elapsed_time()
            })

            if step % cfg.log_every == 0 and step > 0:
                print(stats.get_summary("train"))
                stats.write_tensorboard(summary_writer, "train")
                stats.reset_buffers()
                summary_writer.flush()

            if step % cfg.val_every == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    output = None
                    cfg.visualize(model, output, train_vars, step, epoch, summary_writer, visualization_dir)

                # evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid",
                #          stats, epoch, step, summary_writer, visualization_dir)

            if not debug and step % cfg.ckpt_every == 0 and step > 0:
                utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, stats,
                                     train_vars)

        # Evaluate on the valid split
        if epoch % 5 == 0:
            evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid",
                     stats, epoch, step, summary_writer, visualization_dir)


def evaluate(cfg, model, device, loss_fns, vars, dataloader, split, stats, epoch, step, summary_writer,
             visualization_dir):
    print(f"Evaluate on: {split}")
    if len(dataloader) == 0:
        print("len(dataloader)=0")
        return

    timer = Timer()
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            model_args = [data[arg].to(device) for arg in cfg.model_args]
            labels = data["label"].to(device) if "label" in data else None
            params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

            for i, loss_fn in enumerate(loss_fns, 1):
                output = model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)
                stats.update_stats_to_print(split, loss_dict.keys())
                stats.update(split, step, epoch, {**loss_dict})

        # Visualization
        output = None
        cfg.visualize(model, output, vars, step, epoch, summary_writer, visualization_dir, split)

        # Reconstruction error
        # TODO hack: add temporary model args to dataloader.dataset
        tmp_model_args = ["commands_grouped", "args_grouped"]
        if type(dataloader.dataset) == ConcatDataset:
            datasets_to_hack = dataloader.dataset.datasets
        else:
            datasets_to_hack = [dataloader.dataset]
        for ds in datasets_to_hack:
            ds.model_args = list(ds.model_args) + tmp_model_args

        loss_dict = reconstruction_loss_for_svg_sampled_points(model.module, cfg, dataloader)

        stats.update_stats_to_print(split, loss_dict.keys())
        stats.update(split, step, epoch, {**loss_dict})

        # TODO hack: remove tmp_model_args
        for ds in datasets_to_hack:
            ds.model_args = ds.model_args[:-len(tmp_model_args)]

    stats.update(split, step, epoch, {
        **weights_dict,
        "time": timer.get_elapsed_time()
    })

    print(stats.get_summary(split))
    stats.write_tensorboard(summary_writer, split)
    stats.reset_buffers()
    summary_writer.flush()


def reconstruction_loss_for_svg_sampled_points(model, cfg, dataloader, show_logs=False):
    losses_dict = defaultdict(list)
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            res = _batch_reconstruction_loss_for_svg_sampled_points(model, cfg, data, show_logs)
            for k, v in res.items():
                losses_dict[k].extend(v)

    loss_dict = {}
    for k, losses in losses_dict.items():
        df = pd.DataFrame(losses).describe().transpose()
        df_keys = list(df.keys())
        print(df)
        loss_dict.update({k + "_" + df_key: float(df[df_key]) for df_key in df_keys})
    return loss_dict


def _batch_reconstruction_loss_for_svg_sampled_points(model, cfg, batch, show_logs=False,
                                                      show_logs_images_max_count=None):
    device = next(model.parameters()).device
    commands = batch["commands_grouped"]
    args = batch["args_grouped"]
    losses = defaultdict(list)
    for i, (c, a) in enumerate(zip(commands, args)):  # TODO slow, but not possible to work with batches at the moment
        # target points
        tensor_target = SVGTensor.from_cmd_args(c[0], a[0]).copy().unpad().drop_sos()  # TODO copy problems with device
        points_target = tensor_target.sample_points(n=cfg.n_recon_points)

        # greedy sample prediction
        model_args = [batch[arg][i].unsqueeze(0).to(device) for arg in cfg.model_args]
        z = model.forward(*model_args, encode_mode=True)

        commands_y, args_y = model.greedy_sample(*model_args, z=z)

        try:
            # prediction points
            tensor_pred = SVGTensor.from_cmd_args(commands_y[0], args_y[0])
            points_pred = tensor_pred.sample_points(n=cfg.n_recon_points)
        except Exception as e:
            # print(f"TRY-CATCH, caught exception: {e}")
            continue

        points_target = points_target.to(points_pred.device)

        # reconstruction loss
        for k, loss_recon_fn in cfg.loss_recon_fn_dict.items():
            losses[k].append(loss_recon_fn(points_pred / 256, points_target / 256).item())

        if show_logs:
            print(f"Reconstruction losses")
            for k in cfg.loss_recon_fn_dict.keys():
                print(f"LOSS {i:04} {k}:\t{losses[k][-1]}")

            if show_logs_images_max_count is None or i < show_logs_images_max_count:
                print(f"TARGET points (len={len(points_target)}) vs PREDICTION points (len={len(points_pred)})")
                points_pred_shifted = torch.stack([points_pred[:, 0] + 256, points_pred[:, 1]], dim=1)
                plot_points(torch.cat([points_target, points_pred_shifted]), show_color=True)
                plt.show()

                print(f"TARGET vs PREDICTION with control points")
                make_grid([
                    SVG.from_tensor(tensor_target.data, viewbox=Bbox(256)).split_paths().set_color("random")
                    , SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256)).split_paths().set_color("random")
                ], num_cols=2, grid_width=256).draw_colored(with_points=True)

                print("TARGET:")
                SVG.from_tensor(tensor_target.data, viewbox=Bbox(256)).normalize().split_paths().set_color(
                    "random").draw()
                print("PREDICTION:")
                SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256)).normalize().split_paths().set_color(
                    "random").draw()

    return losses


if __name__ == "__main__":
    print(get_str_formatted_time())
    parser = argparse.ArgumentParser(description='DeepSVG Trainer')
    parser.add_argument("--config-module", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=72)

    args = parser.parse_args()
    utils.set_seed(args.seed)

    cfg = importlib.import_module(args.config_module).Config()
    model_name, experiment_name = args.config_module.split(".")[-2:]

    train(cfg, model_name, experiment_name, log_dir=args.log_dir, debug=args.debug, resume=args.resume,
          eval_only=args.eval_only)
