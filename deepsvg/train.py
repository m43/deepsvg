import argparse
import importlib
import io
import os
from collections import defaultdict

import cairosvg
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn import functional as F
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
from deepsvg.utils.stats import MetricTracker


def train(
        cfg: _Config,
        model_name,
        experiment_name="",
        experiment_identifier=None,
        log_dir="./logs",
        debug=False,
        resume=False,
        eval_only=False,
        eval_l1_loss=True,
        eval_l1_loss_viewbox=24,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Parameters")
    cfg.print_params()
    print("Model Configuration:")
    for key in dir(cfg.model_cfg):
        if not key.startswith("__") and not callable(getattr(cfg.model_cfg, key)):
            print(f"  {key} = {getattr(cfg.model_cfg, key)}")

    print("Loading dataset")
    dataset_load_function = importlib.import_module(cfg.dataloader_module).load_dataset
    dataset_subsets = dataset_load_function(cfg)
    if len(dataset_subsets) == 2:
        train_dataset, valid_dataset = dataset_subsets
        test_dataset = None
    elif len(dataset_subsets) == 3:
        train_dataset, valid_dataset, test_dataset = dataset_subsets
    else:
        raise RuntimeError(f"dataloader_module should return either 2 or 3 subsets,"
                           f"but got {len(dataset_subsets)} subsets")
    print(f"len(train_dataset)={len(train_dataset)}")
    print(f"len(valid_dataset)={len(valid_dataset)}")
    if test_dataset is not None:
        print(f"len(test_dataset)={len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                  num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                                  num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                                     num_workers=cfg.loader_num_workers, collate_fn=cfg.collate_fn)

    model = cfg.make_model().to(device)
    if cfg.pretrained_path is not None:
        print(f"Loading pretrained model {cfg.pretrained_path}")
        utils.load_model(cfg.pretrained_path, model, device)

    stats = Stats(num_steps=cfg.num_steps, num_epochs=cfg.num_epochs, steps_per_epoch=len(train_dataloader),
                  stats_to_print=cfg.stats_to_print)
    train_vars = TrainVars()
    valid_vars = TrainVars()
    test_vars = TrainVars()
    timer = Timer()

    stats.num_parameters = utils.count_parameters(model)
    print(f"#Parameters: {stats.num_parameters:,}")

    # Summary Writer
    current_time = utils.get_str_formatted_time()
    if experiment_identifier is None:
        experiment_identifier = f"{model_name}__{experiment_name}__{current_time}"

    summary_writer = SummaryWriter(
        os.path.join(log_dir, "tensorboard", "debug" if debug else "full", experiment_identifier))
    checkpoint_dir = os.path.join(log_dir, "models", model_name, experiment_identifier)
    visualization_dir = os.path.join(log_dir, "visualization", model_name, experiment_identifier)
    utils.ensure_dir(visualization_dir)

    print("-" * 72)
    print(f"experiment_identifier={experiment_identifier}")
    print(f"summary_writer.logdir={summary_writer.logdir}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"visualization_dir={visualization_dir}")
    print("-" * 72)

    cfg.set_train_vars(train_vars, train_dataloader)
    cfg.set_train_vars(valid_vars, valid_dataloader)
    if test_dataset is not None:
        cfg.set_train_vars(test_vars, test_dataloader)

    # Optimizer, lr & warmup schedulers
    optimizers = cfg.make_optimizers(model)
    scheduler_lrs = cfg.make_schedulers(optimizers, epoch_size=len(train_dataloader))
    scheduler_warmups = cfg.make_warmup_schedulers(optimizers, scheduler_lrs)

    loss_fns = [l.to(device) for l in cfg.make_losses()]
    if resume:
        ckpt_exists = utils.load_ckpt_list(checkpoint_dir, model, device, None, optimizers, scheduler_lrs,
                                           scheduler_warmups, stats, train_vars)
        assert ckpt_exists, f"Could not resume from checkpoint_dir={checkpoint_dir}"

        # Hack to include the additional stats that were not saved in the checkpoint
        for split in cfg.stats_to_print.keys():
            if split not in stats.stats_to_print:
                stats.stats[split] = defaultdict(MetricTracker)
                stats.stats_to_print[split] = set()

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
        if test_dataset is not None:
            evaluate(cfg, model, device, loss_fns, valid_vars, test_dataloader, "test", stats, 0, 0, summary_writer,
                     visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)
            summary_writer.flush()
        evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, 0, 0, summary_writer,
                 visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)
        summary_writer.flush()
        evaluate(cfg, model, device, loss_fns, valid_vars, train_dataloader, "train", stats, 0, 0, summary_writer,
                 visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)
        summary_writer.flush()
        return

    if stats.epoch == 0:
        evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, 0, 0, summary_writer,
                 visualization_dir, eval_l1_loss, eval_l1_loss_viewbox, eval_number_of_batches=3)

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

            # if not debug and step % cfg.ckpt_every == 0 and step > 0:
            #     utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, stats,
            #                          train_vars)

        # Evaluate on the valid split
        if (epoch + 1) % 5 == 0:
            print(f"evaluate on valid, step={step}")
            utils.save_ckpt_list(checkpoint_dir, model, cfg, optimizers, scheduler_lrs, scheduler_warmups, stats,
                                 train_vars)
            evaluate(cfg, model, device, loss_fns, valid_vars, valid_dataloader, "valid", stats, epoch, step,
                     summary_writer, visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)

    if test_dataset is not None:
        # TODO possibly add early stopping and then load the best model here
        print("Test set evaluated with last (not best) checkpoint!")
        evaluate(cfg, model, device, loss_fns, test_vars, test_dataloader, "test", stats, 0, 0, summary_writer,
                 visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)

    print("Train set evaluated with last (not best) checkpoint!")
    evaluate(cfg, model, device, loss_fns, train_vars, train_dataloader, "train", stats, 0, 0, summary_writer,
             visualization_dir, eval_l1_loss, eval_l1_loss_viewbox)


def evaluate(cfg, model, device, loss_fns, vars, dataloader, subset, stats, epoch, step, summary_writer,
             visualization_dir, eval_l1_loss, eval_l1_loss_viewbox, eval_number_of_batches=-1):
    print(f"Evaluate on: {subset}")
    if len(dataloader) == 0:
        print("len(dataloader)=0")
        return

    timer = Timer()
    model.eval()
    with torch.no_grad():
        # Visualization
        output = None
        cfg.visualize(model, output, vars, step, epoch, summary_writer, visualization_dir, subset)

        # Reconstruction error
        # TODO hack: add temporary model args to dataloader.dataset
        tmp_model_args = ["commands_grouped", "args_grouped"]
        if type(dataloader.dataset) == ConcatDataset:
            datasets_to_hack = dataloader.dataset.datasets
        else:
            datasets_to_hack = [dataloader.dataset]
        for ds in datasets_to_hack:
            ds.model_args = list(ds.model_args) + tmp_model_args
        loss_dict = reconstruction_loss_for_svg_sampled_points(
            step=step,
            subset=subset,
            model=model.module,
            cfg=cfg,
            dataloader=dataloader,
            summary_writer=summary_writer,
            eval_l1_loss=eval_l1_loss,
            eval_number_of_batches=eval_number_of_batches,
            eval_l1_loss_viewbox=eval_l1_loss_viewbox,
        )
        stats.update_stats_to_print(subset, loss_dict.keys())
        stats.update(subset, step, epoch, {**loss_dict})

        # TODO hack: remove tmp_model_args
        for ds in datasets_to_hack:
            ds.model_args = ds.model_args[:-len(tmp_model_args)]

        for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
            if batch_idx == eval_number_of_batches:
                break
            model_args = [data[arg].to(device) for arg in cfg.model_args]
            labels = data["label"].to(device) if "label" in data else None
            params_dict, weights_dict = cfg.get_params(step, epoch), cfg.get_weights(step, epoch)

            for i, loss_fn in enumerate(loss_fns, 1):
                output = model(*model_args, params=params_dict)
                loss_dict = loss_fn(output, labels, weights=weights_dict)
                stats.update_stats_to_print(subset, loss_dict.keys())
                stats.update(subset, step, epoch, {**loss_dict})

    stats.update(subset, step, epoch, {
        **weights_dict,
        "time": timer.get_elapsed_time()
    })

    print(stats.get_summary(subset))
    stats.write_tensorboard(summary_writer, subset)
    stats.reset_buffers()
    summary_writer.flush()


def reconstruction_loss_for_svg_sampled_points(step, subset, model, cfg, dataloader, summary_writer,
                                               show_logs=False, eval_l1_loss=True, eval_number_of_batches=-1,
                                               eval_l1_loss_viewbox=24):
    losses_dict = defaultdict(list)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm.tqdm(dataloader)):
            if batch_idx == eval_number_of_batches:
                break
            res = _batch_reconstruction_loss_for_svg_sampled_points(
                step=step,
                batch_idx=batch_idx,
                subset=subset,
                model=model,
                cfg=cfg,
                batch=data,
                summary_writer=summary_writer,
                show_logs=show_logs,
                eval_l1_loss=eval_l1_loss,
                eval_l1_loss_viewbox=eval_l1_loss_viewbox,
            )
            for k, v in res.items():
                losses_dict[k].extend(v)

    loss_dict = {}
    for k, losses in losses_dict.items():
        df = pd.DataFrame(losses).describe().transpose()
        df_keys = list(df.keys())
        print(df)
        loss_dict.update({k + "_" + df_key: float(df[df_key]) for df_key in df_keys})
    return loss_dict


def _batch_reconstruction_loss_for_svg_sampled_points(
        step,
        batch_idx,
        subset,
        model,
        cfg,
        batch,
        summary_writer,
        show_logs=False,
        show_logs_images_max_count=None,
        eval_l1_loss=True,
        eval_l1_loss_resolutions=(128,),  # (64, 128, 256, 512),
        eval_l1_loss_viewbox=24,
):
    device = next(model.parameters()).device
    commands = batch["commands_grouped"]
    args = batch["args_grouped"]
    losses = defaultdict(list)
    n_log_images_in_grid = 8
    log_images = defaultdict(list)
    for i, (c, a) in enumerate(zip(commands, args)):
        # TODO slow, but not possible to work with batches at the moment
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
            points_target = points_target.to(points_pred.device)
        except Exception as e:
            # print(f"TRY-CATCH, caught exception: {e}")
            continue

        svg_target = SVG.from_tensor(tensor_target.data, viewbox=Bbox(256)).split_paths()
        svg_pred = SVG.from_tensor(tensor_pred.data, viewbox=Bbox(256)).split_paths()

        # pointcloud reconstruction loss
        for k, loss_recon_fn in cfg.loss_recon_fn_dict.items():
            losses[f"loss_reconstruction/{k}"].append(loss_recon_fn(points_pred / 256, points_target / 256).item())

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
                    svg_target.set_color("random"),
                    svg_pred.set_color("random"),
                ], num_cols=2, grid_width=256).draw_colored(with_points=True)

                print("TARGET:")
                svg_target.normalize().set_color("random").draw()
                print("PREDICTION:")
                svg_pred.normalize().set_color("random").draw()

        # raster reconstruction loss
        if eval_l1_loss:
            svg_target.normalize(Bbox(eval_l1_loss_viewbox))
            svg_pred.normalize(Bbox(eval_l1_loss_viewbox))
            for res in eval_l1_loss_resolutions:
                rendered_image_pred = svgtensor_to_img(svg_pred, res, res)
                rendered_image_target = svgtensor_to_img(svg_target, res, res)
                l1_loss = F.l1_loss(rendered_image_pred, rendered_image_target)
                losses[f'loss_images_{subset}_{res}x{res}_l1loss_viewbox={eval_l1_loss_viewbox}'].append(l1_loss.item())
                if i < n_log_images_in_grid:
                    log_images[res].append((rendered_image_pred[:, None], rendered_image_target[:, None]))

    if eval_l1_loss and batch_idx % 50 == 0:
        n_columns = int((n_log_images_in_grid * 2) ** 0.5)
        for res in eval_l1_loss_resolutions:
            if len(log_images[res]) == 0:
                continue
            zipped = torch.concat(
                list(
                    map(
                        torch.cat,
                        log_images[res]
                    )
                )
            )
            zipped_image = torchvision.utils.make_grid(zipped, nrow=n_columns)
            summary_writer.add_image(
                f'Images_{subset}_{res}x{res}/l1loss-{batch_idx}-img',
                zipped_image,
                step
            )

    return losses


def svgtensor_to_img(svg, output_width=64, output_height=64):
    pil_image = draw_svgtensor(svg, output_width, output_height)
    return torch.tensor(np.array(pil_image).transpose(2, 0, 1)[-1:]) / 255.


def draw_svgtensor(
        svg,
        output_width,
        output_height,
        fill=False,
        with_points=False,
        with_handles=False,
        with_bboxes=False,
        with_markers=False,
        color_firstlast=False,
        with_moves=True,
):
    svg_str = svg.to_str(fill=fill, with_points=with_points, with_handles=with_handles, with_bboxes=with_bboxes,
                         with_markers=with_markers, color_firstlast=color_firstlast, with_moves=with_moves)

    img_data = cairosvg.svg2png(
        bytestring=svg_str,
        invert_images=True,
        output_width=output_width,
        output_height=output_height,
    )
    return Image.open(io.BytesIO(img_data))


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
