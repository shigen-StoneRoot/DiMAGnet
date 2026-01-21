# DiMAGnet: Multimodal Structure-Guided Diffusion Model for Magnetic Particle Imaging Reconstruction (Official Code)

📌 This repository contains the official implementation of the paper:

📝 "Multimodal Structure-Guided Diffusion Model for Magnetic Particle Imaging Reconstruction"

🧠 We propose a structure-guided multimodal diffusion framework for MPI reconstruction, where structural information is integrated to guide the reconstruction process and improve image fidelity and robustness.

> ⚠️ Status: The manuscript is currently under **major revision** at **Medical Image Analysis (MedIA)**.

---

## Table of Contents

- 📚 [Citation](#citation)
- ⚙️ [1. Environment Setup](#1-environment-setup)
- 🗂️ [2. Data Preparation](#2-data-preparation)
- 🧪 [3. OpenMPI Reconstruction](#3-openmpi-reconstruction)
- 🚀 [4. Multimodal Training and Inference](#4-multimodal-training-and-inference)
- 🧩 [5. Single-Modal Training and Inference (TODO)](#5-single-modal-training-and-inference-todo)

---

## Citation
📝 TODO (Under Review in Journal "Medical Image Analysis")

---

## 1. Environment Setup

🔧 Our method is based on improved-diffusion. For building the environment, please refer to https://github.com/openai/improved-diffusion.

📦 Additionally, we require some supplementary environments, including:

~~~text
mpi4py == 4.1.1
numpy == 1.24.4
scikit-image == 0.21.0
scipy == 1.10.1
monai == 1.3.2
~~~

---

## 2. Data Preparation

### 2.1 OpenMPI Data

1. 📥 Download the OpenMPI dataset from the provided Google Drive folder:
   - 🔗 Google Drive link: https://drive.google.com/drive/folders/1xyl-oVO8HXyZyggA48kiicmUgCYJWfY4?usp=drive_link

2. 📦 Unzip (extract) the downloaded files, and place the extracted folder into:

   - 📁 ./Data/OpenMPI#2/

Your directory should look like:

./Data/
  OpenMPI#2/
    ... (OpenMPI files)

> ⚠️ Note: the folder name contains a `#` (OpenMPI#2). Please keep it exactly the same as required by the code.

---

### 2.2 Multimodal Simulation Dataset (MM_dataset)

📁 For the multimodal simulated dataset, please create two subfolders under ./Data/MM_dataset/:

- ✅ ./Data/MM_dataset/images/
- ✅ ./Data/MM_dataset/signals/

🧾 Both folders should contain .npy files, and filenames must match one-to-one (same base name), e.g.:

~~~text
./Data/MM_dataset/
  images/
    case_0001.npy
    case_0002.npy
    ...
  signals/
    case_0001.npy
    case_0002.npy
    ...
~~~

#### images/*.npy format

- Shape: (100, 4, 64, 64)
  - 100: number of samples per case (we randomly generate 100 samples for each case)
  - 4: four channels
  - 64 x 64: spatial resolution

Channel definition (in order):

1) $S^{\dagger} u_{clean}$
2) $S^{\dagger} u_{noisy}$
3) $c_{mpi}$
4) $c_{ref}$

#### signals/*.npy format

- Shape: (100, 2, M)
  - 100: number of samples per case (aligned with images)
  - 2: two signal channels
  - M: signal length

Channel definition (in order):

1) $u_{noisy}$
2) $u_{clean}$

---

### 2.3 System Matrix Preprocessing

⭐ Before building the dataset, we strongly recommend denoising the measured system matrix (SM) using our previous work **U-N2C**:
- 🔗 https://github.com/WrinkleXuan/U-N2C

This step typically improves the stability of the pseudo-inverse and helps produce cleaner back-projected images.

> ✅ Before computing the pseudo-inverse, please apply **energy normalization** to the system matrix to avoid scale issues.

We use a Tikhonov-regularized inverse to compute the pseudo-inverse $S^{\dagger}$. The code is:

~~~python
def tikhonov_inverse(S, lamb=1e-2):
    SH = S.conj().T
    regularizer = lamb * np.eye(S.shape[1])
    inverse = np.linalg.inv(SH @ S + regularizer) @ SH
    return inverse
~~~


Please tune the lamb to balance noise suppression and detail preservation. As a practical guideline, try lamb in [1e-3, 100].

We recommend selecting lamb such that the back-projected result $S^{\dagger} u_noisy$ is visually recognizable (i.e., not dominated by noise and not over-smoothed), since it is used as an important input channel in the multimodal dataset construction.

---

### 2.4 Important: Real/Imag Concatenation for Frequency Data

⚡ We store frequency-domain data by concatenating the real and imaginary parts along the last dimension:

- first half: real part
- second half: imaginary part

> 📌 This convention applies to both the system matrix and signals.

---

## 3. OpenMPI Reconstruction

▶️ After data preparation, run the following command:

~~~bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/image_sample.py \
  --model_path ./checkpoints/model_ck_OpenMPI.pt \
  --image_size 64 \
  --num_channels 16 \
  --num_res_blocks 2 \
  --diffusion_steps 100 \
  --noise_schedule linear \
  --predict_xstart True \
  > OpenMPI_sampling.out 2>&1 &
~~~

📦 The sampled results will be saved to:

~~~text
./save_data/samples_3x1x64x64.npz
~~~

🖼️ To visualize the generated images, run:

~~~bash
python visualization_openMPI.py
~~~

---

## 4. Multimodal Training and Inference

🧩 This section describes how to train the model and perform inference (reconstruction) using the multimodal simulation dataset.

---

### 4.1 Multimodal Training

🚀 Run the following command to start training:

~~~bash
mpiexec -n 1 python scripts/image_train.py \
  --data_dir ./Data/MM_dataset/images \
  --image_size 64 \
  --num_channels 16 \
  --num_res_blocks 2 \
  --diffusion_steps 100 \
  --noise_schedule linear \
  --lr 1e-2 \
  --batch_size 64 \
  --predict_xstart True \
  --distill False \
  > multimodal_training.out 2>&1 &
~~~

🗃️ Training logs and checkpoints will be saved under ./log_dir/.

---

### 4.2 Multimodal Inference (Reconstruction)

🧪 After training (or using a provided checkpoint), run the following command to reconstruct images:

~~~bash
CUDA_VISIBLE_DEVICES=1 nohup python scripts/image_sample_mm.py \
  --model_path ./log_dir/openai-xxxxxxxx/modelxxxxxx.pt \
  --image_size 64 \
  --num_channels 16 \
  --num_res_blocks 2 \
  --diffusion_steps 100 \
  --noise_schedule linear \
  --predict_xstart True \
  --distill False \
  > multimodal_sampling.out 2>&1 &
~~~

---

## 5. Single-Modal Training and Inference (TODO)
