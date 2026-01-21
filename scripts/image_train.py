"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_mpi_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch as th


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")


    
    train_subs_path = r'./Data/MM_dataset/train_subs.txt'
    if train_subs_path is not None:
        with open(train_subs_path, "r", encoding="utf-8") as file:
            train_subs = [line.strip() for line in file]
    else:
        train_subs = None

    data = load_mpi_data(
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    image_size=args.image_size,
    class_cond=False,
    subs=train_subs
    )

    
    import numpy as np
    project_matrix = np.load(r'./Data/MM_dataset/proj.npy')
    pinv_S = np.load(r'./Data/MM_dataset/pinv_A.npy')

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        total_iters = args.total_iters,
        project_matrix=project_matrix,
        pinv_S=pinv_S,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=100, 
        total_iters=500,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        distill=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
