from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn.functional as F
import torch

def compute_masked_feature(total_feats, threshold=0.535):
    feats_sigmoid = torch.sigmoid(total_feats)
    mask = (feats_sigmoid >= threshold).float()
    feats_masked = feats_sigmoid * mask

    fused_mask1 = feats_masked.sum(dim=1, keepdim=True)  # (B, H, W)
    fused_mask2 = F.interpolate(fused_mask1, scale_factor=0.5, mode='bilinear', align_corners=False)
    fused_mask3 = F.interpolate(fused_mask1, scale_factor=0.25, mode='bilinear', align_corners=False)


    def norm_img(fused_mask):
        min_vals = fused_mask.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = fused_mask.amax(dim=(1, 2, 3), keepdim=True)

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # 避免除0

        fused_mask_norm = (fused_mask - min_vals) / range_vals

        return fused_mask_norm
    
    fused_mask1 = norm_img(fused_mask1)
    fused_mask2 = norm_img(fused_mask2)
    fused_mask3 = norm_img(fused_mask3)
    
    return fused_mask1, fused_mask2, fused_mask3


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
    

def load_mpi_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, subs=None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = MPI_ImageDataset(
        image_size,
        data_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        subs=subs
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, 
            pin_memory=True, prefetch_factor=4, persistent_workers=True
        )

    while True:
        yield from loader

def load_mpi_distil_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, subs=None, 
    distil_path = r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/IXI_train_feats/split_feats', 
    distil_flag=True
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = MPI_Distil_ImageDataset(
        image_size,
        data_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        subs=subs, 
        distil_path=distil_path,
        distil_flag=distil_flag,
    )

    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True, 
            pin_memory=True, prefetch_factor=10, persistent_workers=True,
        )

    while True:
        yield from loader

def load_mpi_valid_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, subs=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = MPI_ImageDataset(
        image_size,
        data_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        subs=subs,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, 
            pin_memory=True, prefetch_factor=8,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, 
            pin_memory=True, prefetch_factor=2,
        )

    return loader

class MPI_ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, signal_paths=None, classes=None, shard=0, num_shards=1, subs=None):
        super().__init__()
        self.resolution = resolution
        # self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        if signal_paths is None:
            self.signal_paths = image_paths.replace('images', 'signals')
            self.STD_paths = image_paths.replace('images', 'signals_mean_std')

        if subs is None:
            self.subs = os.listdir(image_paths)
        else:
            self.subs = subs
        self.local_images = [os.path.join(image_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]
        self.local_signals = [os.path.join(self.signal_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]
        self.local_STDs = [os.path.join(self.STD_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        base, suffix = path.rsplit('_', 1)  # 以'_'为分隔符，从右侧分割一次
        # print(base, suffix)
        suffix = int(suffix)         # 将后缀转换为整数

        # print(base, suffix)

        img = np.load(base)[suffix] #(4, 64, 64) 0 channel -- GT NR; 1 channel -- noisy NR; 2 channel -- MRI ; 3 channel -- MPI

        arr = (img - np.mean(img, (1, 2)).reshape(-1, 1, 1)) / np.maximum(np.std(img, (1, 2)).reshape(-1, 1, 1), 1e-8)
        
        arr[-1, :, :] = img[-1, :, :] / 0.5 - 1 # MPI 缩放到-1， 1

        # arr[0, :, :] = (img[0, :, :] - img[0, :, :].min()) / (img[0, :, :].max() - img[0, :, :].min())
        # arr[1, :, :] = (img[1, :, :] - img[1, :, :].min()) / (img[1, :, :].max() - img[1, :, :].min())

        # arr[0, :, :] = img[0, :, :] / np.abs(img[0, :, :]).max()

        arr[1, :, :] = img[1, :, :] / np.abs(img[1, :, :]).max()

        # arr[1, :, :] = awgn(arr[1, :, :], snr=40)

        # arr[1, :, :] = add_awgn_without_clean(arr[1, :, :], 40, 30)

        # arr[1, :, :] = img[0, :, :] / np.abs(img[0, :, :]).max()
        

        # arr[0, :, :] = img[0, :, :]

        # ### --------------------
        # arr = np.load(base)[suffix] #(4, 64, 64) 0 channel -- GT NR; 1 channel -- noisy NR; 2 channel -- MRI ; 3 channel -- MPI        
        # arr[-1, :, :] = arr[-1, :, :] / 0.5 - 1 # MPI 缩放到-1， 1
        # # arr[:2, :, :] = (arr[:2, :, :] - np.mean(arr[:2, :, :], (1, 2)).reshape(-1, 1, 1)) / np.maximum(np.std(arr[:2, :, :], (1, 2)).reshape(-1, 1, 1), 1e-10) # NR 进行z标准化
        # ### --------------------



        sig_path = self.local_signals[idx]
        sig_base, _ = sig_path.rsplit('_', 1)  # 以'_'为分隔符，从右侧分割一次
        sig = np.load(sig_base)[suffix]

        STD = np.array([1])

        
        arr = arr.astype(np.float32)
        sig = sig.astype(np.float32)
        STD = STD.astype(np.float32)


        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr[:, :, :], sig, STD, out_dict

    
import concurrent.futures
import concurrent
import tqdm


class MPI_Distil_ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, signal_paths=None, classes=None, shard=0, num_shards=1, subs=None, 
                 distil_path=r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/IXI_train_feats/split_feats', distil_flag=True):
        super().__init__()
        self.resolution = resolution
        # self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        if signal_paths is None:
            self.signal_paths = image_paths.replace('images', 'signals')
            self.STD_paths = image_paths.replace('images', 'signals_mean_std')

        if subs is None:
            self.subs = os.listdir(image_paths)
        else:
            self.subs = subs
        self.local_images = [os.path.join(image_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]
        self.local_signals = [os.path.join(self.signal_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]
        self.local_STDs = [os.path.join(self.STD_paths, f"{item}_{i:02}") for item in self.subs for i in range(100)]
        self.distil_path = distil_path
        self.distil_flag = distil_flag


        MASK = np.load(r'./MASK.npy', allow_pickle=True)
        self.MASK1 = MASK[0]
        self.MASK2 = MASK[1]
        self.MASK3 = MASK[2]

    def __len__(self):
        return len(self.local_images)


    def init_mask(self):
        MASK1 = []
        MASK2 = []
        MASK3 = []

        def process_one(idx):
            path = self.local_images[idx]
            base, suffix = path.rsplit('_', 1)
            distil_p = os.path.join(self.distil_path + '_npy', 'x1_mpi_feats', base.rsplit('/', 1)[-1] + '_' + suffix + '.npy')
            feats = np.load(distil_p)
            feats = torch.from_numpy(feats).float().unsqueeze(0).mean(1)
            mask1, mask2, mask3 = compute_masked_feature(feats)
            return mask1, mask2, mask3

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm.tqdm(executor.map(process_one, range(len(self.local_images))), total=len(self.local_images)))
            print(type(results[0]), len(results[0]))
            print(results[0][0].shape)

        # 拆分结果
        for mask1, mask2, mask3 in results:
            
            MASK1.append(mask1)
            MASK2.append(mask2)
            MASK3.append(mask3)

        # cat然后保存
        MASK1 = torch.cat(MASK1, 0).numpy()
        MASK2 = torch.cat(MASK2, 0).numpy()
        MASK3 = torch.cat(MASK3, 0).numpy()

        print(MASK1.shape)

        MASK_arr = np.empty(3, dtype=object)
        MASK_arr[0] = MASK1
        MASK_arr[1] = MASK2
        MASK_arr[2] = MASK3

        np.save(r'./MASK.npy', MASK_arr)
        print('finished mask construction')

    def __getitem__(self, idx):
        path = self.local_images[idx]
        base, suffix = path.rsplit('_', 1)  # 以'_'为分隔符，从右侧分割一次
        
        feats = np.load(os.path.join(self.distil_path + '_npy', 'ALL', base.rsplit('/', 1)[-1] + '_' + suffix + '.npy'), allow_pickle=True)

        if feats[0].shape[0] == 1:
            feats = [np.repeat(feats[i], 7, axis=0) for i in range(len(feats))]

        if self.distil_flag:

            distil_feats_dict = {

                'x1_mpi_feats': feats[0].astype(np.float32), 
                'x2_mpi_feats': feats[1].astype(np.float32),
                'x3_mpi_feats': feats[2].astype(np.float32),
                
                'x1_fusion_feats': feats[3].astype(np.float32), 
                'x2_fusion_feats': feats[4].astype(np.float32), 
                'x3_fusion_feats': feats[5].astype(np.float32), 

                'x2_out_feats': feats[6].astype(np.float32),
                'x3_out_feats': feats[7].astype(np.float32),
                'x4_out_feats': feats[8].astype(np.float32),

                'logits': feats[9].astype(np.float32),

                'mask1':self.MASK1[idx], 
                'mask2':self.MASK2[idx], 
                'mask3':self.MASK3[idx],
            }

        else:
            distil_feats_dict = {}
        suffix = int(suffix)         # 将后缀转换为整数

        


        img = np.load(base, mmap_mode='r')[suffix] #(4, 64, 64) 0 channel -- GT NR; 1 channel -- noisy NR; 2 channel -- MRI ; 3 channel -- MPI

        arr = img.copy() #(4, 64, 64) 0 channel -- GT NR; 1 channel -- noisy NR; 2 channel -- MRI ; 3 channel -- MPI        

        arr[:2, :, :] = (arr[:2, :, :] - np.mean(arr[:2, :, :], (1, 2)).reshape(-1, 1, 1)) / np.std(arr[:2, :, :], (1, 2)).reshape(-1, 1, 1) # NR 进行z标准化


        sig_path = self.local_signals[idx]
        sig_base, _ = sig_path.rsplit('_', 1)  # 以'_'为分隔符，从右侧分割一次
        sig = np.load(sig_base)[suffix]

        STD_path = self.local_STDs[idx]
        STD_base, _ = STD_path.rsplit('_', 1)  # 以'_'为分隔符，从右侧分割一次
        STD = np.load(STD_base)[suffix]

        
        arr = arr.astype(np.float32)
        sig = sig.astype(np.float32)
        STD = STD.astype(np.float32)


        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr[:, :, :], sig, STD, out_dict, distil_feats_dict



class CustomeDataset(Dataset):
    def __init__(self, resolution, image_data, signal_data=None, classes=None, shard=0, num_shards=1, real_data=False):
        super().__init__()
        self.resolution = resolution
        # self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.signal_data = signal_data
        self.image_data = image_data
        self.real_data = real_data
    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, idx):

        img = self.image_data[idx] # 0-noisyNR, 1 -- GTNR, 2 -- MPI, 3 -- MRI, 4 -- mask
        sig = self.signal_data[idx]


        img = (img - np.mean(img, (1, 2))[:, np.newaxis, np.newaxis]) / np.std(img, (1, 2))[:, np.newaxis, np.newaxis]




        arr = img.astype(np.float32)
        sig = sig.astype(np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr[:, :, :], sig, out_dict
    


def load_custome_valid_data(
    img_data, signal_data, batch_size, image_size, class_cond=False, deterministic=True, real_data=False,
):
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = CustomeDataset(
        resolution=image_size,
        image_data=img_data,
        signal_data=signal_data,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        real_data = real_data,
    )

    loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False
        )

    return loader
    
if __name__ == '__main__':

    # img_data = np.load(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/noise_analysis/total_imgs_30dB.npy')
    # sig_data = np.load(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/noise_analysis/total_signals_30dB.npy')

    # sample_loader = load_custome_valid_data(img_data=img_data, signal_data=sig_data, batch_size=512, image_size=64)

    # for (batch, sig, cond) in sample_loader:
    #     noisy_NR = batch[:, 0:1, :, :]               # null range
    #     structure_condition = batch[:, 1:2, :, :] # MRI
    #     batch = batch[:, 2:3, :, :]               # MPI

    #     import matplotlib.pyplot as plt
    #     plt.subplot(131)
    #     plt.imshow(noisy_NR[1].cpu().squeeze())
    #     plt.subplot(132)
    #     plt.imshow(structure_condition[1].cpu().squeeze())
    #     plt.subplot(133)
    #     plt.imshow(batch[1].cpu().squeeze())
    #     plt.savefig("./1.jpg")
    #     plt.show()

    #     print(noisy_NR.shape, structure_condition.shape, batch.shape)

    #     print(noisy_NR[1].min(), noisy_NR[1].max())
    #     print(structure_condition[1].min(), structure_condition[1].max())
    #     print(batch[1].min(), batch[1].max())
    #     break


    # dataset = MPIImageDataset(64, r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion', classes=None)
    # print(dataset.__len__)
    # arr, out_dict = dataset.__getitem__(10)
    # print(arr.shape)

    # data = load_mpi_data(
    #     data_dir=r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/images',
    #     batch_size=32,
    #     image_size=64,
    #     class_cond=False
    # )

    # import torch
    # batch, sig, STD, cond = next(data)
    # print(batch.shape, sig.shape, STD.shape)
    MASK = np.load(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/IXI_train_feats/MASK.npy', allow_pickle=True)
    print(MASK[0].shape)
    

    train_subs_path = r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/train_subs.txt'
    if train_subs_path is not None:
        with open(train_subs_path, "r", encoding="utf-8") as file:
            train_subs = [line.strip() for line in file]
    else:
        train_subs = None

    data = load_mpi_distil_data(
    data_dir=r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/images',
    batch_size=16,
    image_size=64,
    class_cond=False,
    subs=train_subs,
    distil_path=r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/IXI_train_feats/split_feats',
    distil_flag=True
    )


    import torch
    batch, sig, STD, cond, distil_feat = next(data)
    print(batch.shape, sig.shape, STD.shape)
    # print(distil_feat.keys())

    def compute_masked_feature(total_feats, threshold=0.535):
        feats_sigmoid = torch.sigmoid(total_feats)
        mask = (feats_sigmoid >= threshold).float()
        feats_masked = feats_sigmoid * mask

        fused_mask1 = feats_masked.sum(dim=1, keepdim=True)  # (B, H, W)
        fused_mask2 = F.interpolate(fused_mask1, scale_factor=0.5, mode='bilinear', align_corners=False)
        fused_mask3 = F.interpolate(fused_mask1, scale_factor=0.25, mode='bilinear', align_corners=False)


        def norm_img(fused_mask):
            min_vals = fused_mask.amin(dim=(1, 2, 3), keepdim=True)
            max_vals = fused_mask.amax(dim=(1, 2, 3), keepdim=True)

            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1.0  # 避免除0

            fused_mask_norm = (fused_mask - min_vals) / range_vals

            return fused_mask_norm
        
        fused_mask1 = norm_img(fused_mask1)
        fused_mask2 = norm_img(fused_mask2)
        fused_mask3 = norm_img(fused_mask3)
        
        return fused_mask1, fused_mask2, fused_mask3
    

    GT_img = batch[:, 3:4, :, :]
    fusion_feat = distil_feat['logits']
    print(fusion_feat.shape)
    # feat_mask1, feat_mask2, feat_mask3 = compute_masked_feature(fusion_feat.mean(2), threshold=0.52)
    # print(feat_mask1.shape, feat_mask2.shape, feat_mask3.shape)

    # import matplotlib.pyplot as plt
    # idx = 2
    # plt.subplot(221)
    # plt.imshow(GT_img[idx].squeeze().numpy())
    # plt.subplot(222)
    # plt.imshow(feat_mask1[idx].squeeze().numpy(), cmap='jet')
    # plt.subplot(223)
    # plt.imshow(feat_mask2[idx].squeeze().numpy(), cmap='jet')
    # plt.subplot(224)
    # plt.imshow(feat_mask3[idx].squeeze().numpy(), cmap='jet')
    # # plt.colorbar()
    # plt.savefig("./00distil.jpg")

    # plt.show()

    # for k in distil_feat.keys():
    #     print(k, distil_feat[k].shape)
    # GT_img = batch[:, 3:4, :, :]
    # GT_img = (GT_img + 1) / 2

    # mask = GT_img > 0

    # import matplotlib.pyplot as plt
    # idx = 10
    # plt.subplot(121)
    # plt.imshow(GT_img[idx].cpu().squeeze().numpy())

    # plt.subplot(122)
    # plt.imshow(mask[idx].cpu().squeeze().numpy())

    # plt.savefig("./00distil.jpg")
    # plt.show()

    # exit()

    # noisy_sig = sig[:, 0:1, :]
    # sig_mean, sig_std = STD[:, 0:1, :], STD[:, 1:2, :]
    # noisy_sig = noisy_sig * sig_std + sig_mean
    # noisy_sig = noisy_sig / torch.amax(noisy_sig, dim=-1, keepdim=True)

    # print(noisy_sig.shape)

    # for batch in data:
    #     print(batch[0].shape)
    #     print(batch[0].max(), batch[0].min())
        
    #     break

    # print(batch.shape)
    # print(sig.shape)
    # print(STD.shape)

    # import pickle

    # img_data = pickle.load(open(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/ADMM/total_img_data.pkl', 'rb'))
    # sig_data = pickle.load(open(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/ADMM/signals_40dB.pkl', 'rb'))
    
    # dataloader = load_custome_valid_data(img_data=img_data, signal_data=sig_data, batch_size=512, image_size=64)

    # import matplotlib.pyplot as plt
    # for img, sig, _ in dataloader:
    #     print(img.shape, sig.shape)
    #     print(img[0, 3:4, :, :].min(), img[0, 3:4, :, :].max())
    #     mri = img[30, 4:5, :, :].numpy().squeeze()
    #     plt.imshow(mri)
    #     plt.savefig("./1.jpg")
    #     plt.show()


    #     break

    # import pickle
    # img_data, sig_data = pickle.load(open(r'/data/shigen/Diffusion_datasets/MPI_MRI_fusion/Real_Data/tree2.pkl', 'rb'))
    # print(sig_data.shape)
    # sig_data = sig_data[np.newaxis, :, :]
    # sample_loader = load_custome_valid_data(img_data=img_data, signal_data=sig_data, batch_size=1, image_size=64, real_data=True)
    # for img, sig, _ in sample_loader:
    #     print(img.shape, sig.shape)


