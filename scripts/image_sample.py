"""
Generate image samples from a diffusion model and save them as a numpy array.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import tqdm

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Ensure local project has priority over other folders with same package name.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_valid_subs(txt_path: str):
    if not txt_path:
        return None
    with open(txt_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_openmpi_loaders():
    """Build two dataloaders for OpenMPI pickles (as in your current script)."""
    from improved_diffusion.image_datasets import load_custome_valid_data
    import pickle

    base = "./Data/OpenMPI#2"
    img1, sig1 = pickle.load(open(os.path.join(base, "resolution_OpenMPI.pkl"), "rb"))
    img2, sig2 = pickle.load(open(os.path.join(base, "shape_OpenMPI.pkl"), "rb"))
    img3, sig3 = pickle.load(open(os.path.join(base, "concentration_OpenMPI.pkl"), "rb"))

    sig1 = sig1[np.newaxis, :, :]  # (1,1198,?)
    sig2 = sig2[np.newaxis, :, :]
    sig3 = sig3[np.newaxis, :, :]

    img_data_12 = np.concatenate([img1, img2], axis=0)
    sig_data_12 = np.concatenate([sig1, sig2], axis=0)

    loader_main = load_custome_valid_data(
        img_data=img_data_12,
        signal_data=sig_data_12,
        batch_size=2,
        image_size=64,
        real_data=True,
    )
    loader_con = load_custome_valid_data(
        img_data=img3,
        signal_data=sig3,
        batch_size=1,
        image_size=64,
        real_data=True,
    )
    return loader_main, loader_con


def load_projection_mats():
    base = "./Data/OpenMPI#2"
    proj = th.from_numpy(np.load(os.path.join(base, "proj_4096x4096_real.npy")))
    pinv = th.from_numpy(np.load(os.path.join(base, "pinv_A_4096x1198_real.npy")))

    proj_l7 = th.from_numpy(np.load(os.path.join(base, "proj_4096x4096_real_layer7.npy")))
    pinv_l7 = th.from_numpy(np.load(os.path.join(base, "pinv_A_4096x1198_real_layer7.npy")))

    return (proj, pinv), (proj_l7, pinv_l7)


@th.no_grad()
def sample_from_loader(
    *,
    model,
    diffusion,
    data_loader,
    image_size: int,
    clip_denoised: bool,
    class_cond: bool,
    use_ddim: bool,
    project_matrix: th.Tensor,
    device: th.device,
):
    """
    Run sampling on a dataloader that yields (batch, sig, cond).

    Expected tensor layout (as in your script):
      batch: (B,2,64,64) where batch[:,0:1] = noisy_NR, batch[:,1:2] = GT_NR / structure / MPI
      sig:   (B,?,1198)  -> noisy_sig = sig[:,0:1,:]
    """
    all_images = []
    all_labels = []
    all_gt_mpis = []

    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop

    for batch, sig, _cond in data_loader:
        b = batch.shape[0]

        noisy_sig = sig[:, 0:1, :]  

        gt_nr = batch[:, 1:2, :, :]
        noisy_nr = batch[:, 0:1, :, :]
        structure = batch[:, 1:2, :, :]
        gt_mpi = batch[:, 1:2, :, :] 

        condition_kwargs = {
            "structure": structure,
            "project_matrix": project_matrix,
            "noisy_Null_Range": noisy_nr,
            "GT_Null_Range": gt_nr,
            "noisy_sig": noisy_sig,
        }

        model_kwargs = {}
        classes = None
        if class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(b,), device=device
            )
            model_kwargs["y"] = classes

        # store GT MPI (for later metrics)
        all_gt_mpis.append(gt_mpi.cpu().numpy())

        # sample
        sample = sample_fn(
            model,
            (b, 1, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            condition_kwargs=condition_kwargs,
        ).to(th.float).contiguous()[:b]

        # distributed gather
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([s.cpu().numpy() for s in gathered_samples])

        if class_cond and classes is not None:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([l.cpu().numpy() for l in gathered_labels])

        logger.log(f"created {len(all_images) * b} samples (accumulated)")

    arr = np.concatenate(all_images, axis=0)
    gt = np.concatenate(all_gt_mpis, axis=0) if len(all_gt_mpis) else None
    labels = np.concatenate(all_labels, axis=0) if len(all_labels) else None
    return arr, gt, labels


def save_results(arr, gt, labels, save_path: str, class_cond: bool):
    if dist.get_rank() != 0:
        return

    os.makedirs(save_path, exist_ok=True)
    shape_str = "x".join([str(x) for x in arr.shape])

    out_path = os.path.join(save_path, f"samples_{shape_str}.npz")
    gt_path = os.path.join(save_path, f"gt_{shape_str}.npz")

    logger.log(f"saving samples to {out_path}")
    if class_cond and labels is not None:
        np.savez(out_path, arr, labels)
    else:
        np.savez(out_path, arr)

    if gt is not None:
        logger.log(f"saving gt to {gt_path}")
        np.savez(gt_path, gt)


def evaluate_psnr_ssim(arr, gt):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    # [-1,1] range
    psnrs = [psnr(arr[i].squeeze(), gt[i].squeeze(), data_range=2.0) for i in range(gt.shape[0])]
    ssims = [ssim(arr[i].squeeze(), gt[i].squeeze(), data_range=2.0) for i in range(gt.shape[0])]
    logger.log(f"Metrics in [-1,1]:  PSNR={np.mean(psnrs):.4f}, SSIM={np.mean(ssims):.4f}")

    # [0,1] range
    arr01 = (arr + 1) / 2
    gt01 = (gt + 1) / 2
    psnrs = [psnr(arr01[i].squeeze(), gt01[i].squeeze(), data_range=1.0) for i in range(gt01.shape[0])]
    ssims = [ssim(arr01[i].squeeze(), gt01[i].squeeze(), data_range=1.0) for i in range(gt01.shape[0])]
    logger.log(f"Metrics in [0,1]:   PSNR={np.mean(psnrs):.4f}, SSIM={np.mean(ssims):.4f}")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log(f"loading model weights from {args.model_path}")
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    # Load data
    sample_loader, sample_loader2 = build_openmpi_loaders()

    # Load projection matrices (you only use project_matrix and project_matrix2)
    (proj, _pinv), (proj_l7, _pinv_l7) = load_projection_mats()
    proj = proj.to(dist_util.dev())
    proj_l7 = proj_l7.to(dist_util.dev())

    logger.log("sampling...")

    # 1) concentration loader uses layer7 proj (your original logic)
    arr2, gt2, labels2 = sample_from_loader(
        model=model,
        diffusion=diffusion,
        data_loader=sample_loader2,
        image_size=args.image_size,
        clip_denoised=args.clip_denoised,
        class_cond=args.class_cond,
        use_ddim=args.use_ddim,
        project_matrix=proj_l7,
        device=dist_util.dev(),
    )

    # 2) main loader uses base proj
    arr1, gt1, labels1 = sample_from_loader(
        model=model,
        diffusion=diffusion,
        data_loader=sample_loader,
        image_size=args.image_size,
        clip_denoised=args.clip_denoised,
        class_cond=args.class_cond,
        use_ddim=args.use_ddim,
        project_matrix=proj,
        device=dist_util.dev(),
    )

    # Merge results
    arr = np.concatenate([arr2, arr1], axis=0)
    arr = np.tanh(arr * 0.95)
    gt = None if (gt1 is None and gt2 is None) else np.concatenate([gt2, gt1], axis=0)
    labels = None
    if args.class_cond and (labels1 is not None or labels2 is not None):
        labels = np.concatenate([labels2, labels1], axis=0)

    save_results(arr, gt, labels, args.save_path, args.class_cond)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=32,
        batch_size=512,
        use_ddim=False,
        model_path="",
        distill=True,
        save_path="./save_data",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
