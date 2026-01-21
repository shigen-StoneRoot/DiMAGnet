"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion.image_datasets import load_mpi_data, load_mpi_valid_data
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    model.eval()

    valid_subs_path = r'./Data/MM_dataset/test_subs.txt'

    if valid_subs_path is not None:
        with open(valid_subs_path, "r", encoding="utf-8") as file:
            valid_subs = [line.strip() for line in file]
    else:
        valid_subs = None



    sample_loader = load_mpi_valid_data(
    data_dir=r'./Data/MM_dataset/images',
    batch_size=args.batch_size,
    image_size=args.image_size,
    class_cond=False,
    subs=valid_subs
    )


    project_matrix = np.load(r'./Data/MM_dataset/proj.npy')
    project_matrix = th.from_numpy(project_matrix)

    pinv_S = np.load(r'./Data/MM_dataset/pinv_A.npy')
    pinv_S = th.from_numpy(pinv_S).float()


    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_MPIs = []


    for (batch, sig, STD, cond) in sample_loader:
        condition_kwargs = {}
        noisy_sig = sig[:, 0:1, :]

        b = batch.shape[0]

        GT_NR = batch[:, 0:1, :, :]               # null range
        noisy_NR = batch[:, 1:2, :, :]               # null range
        structure_condition = batch[:, 2:3, :, :] # MRI
        batch = batch[:, 3:4, :, :]               # MPI

        condition_kwargs['structure'] = structure_condition
        condition_kwargs['project_matrix'] = project_matrix
        condition_kwargs['noisy_Null_Range'] = noisy_NR
        condition_kwargs['GT_Null_Range'] = GT_NR
        condition_kwargs['noisy_sig'] = None


        all_MPIs.append(batch.cpu().numpy()[:b, :, :, :])

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(b,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )


        sample = sample_fn(
            model,
            (b, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            condition_kwargs=condition_kwargs,
        )


        sample = sample.to(th.float)
        sample = sample.clamp(-1, 1)
        sample = sample.contiguous()[:b, :, :, :]

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)


    if len(all_MPIs) != 0:
        all_MPIs = np.concatenate(all_MPIs, axis=0)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        gt_path = os.path.join(logger.get_dir(), f"gt_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            if len(all_MPIs) != 0:
                np.savez(gt_path, all_MPIs)

    dist.barrier()
    logger.log("sampling complete")


    # def norm_img(img):
    #     return img
    
    # from skimage.metrics import peak_signal_noise_ratio as psnr
    # from skimage.metrics import structural_similarity as ssim

  
    # all_MPIs = (all_MPIs + 1) / 2

    # sample = (arr + 1) / 2

    # all_MPIs = all_MPIs.squeeze()
    # sample = sample.squeeze()

    # mn = np.min(sample, (1, 2))[:, np.newaxis, np.newaxis]
    # mx = np.max(sample, (1, 2))[:, np.newaxis, np.newaxis]
    # sample = (sample - mn) / (mx - mn + 1e-8)


    # print(sample.min(), sample.max())
    # print(all_MPIs.min(), all_MPIs.max())

    # PSNRs = [psnr(norm_img(sample[idx].squeeze()), norm_img(all_MPIs[idx].squeeze()), data_range=1.0) for idx in range(all_MPIs.shape[0])]
    # SSIMs = [ssim(norm_img(sample[idx].squeeze()), norm_img(all_MPIs[idx].squeeze()), data_range=1.0) for idx in range(all_MPIs.shape[0])]

    # print(np.mean(PSNRs), np.mean(SSIMs))
   

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=32,
        batch_size=512,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
