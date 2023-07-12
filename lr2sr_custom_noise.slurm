#!/bin/bash
#SBATCH -p TWIG
#SBATCH -N 1
#SBATCH --job-name=gen-sr
#SBATCH --gpus=a100-40:1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xiaowen4@andrew.cmu.edu

hostname; pwd; date


Boxsize=100000 # Boxsize of LR/HR/SR, in kpc/h
nsplit=4 # split the LR box into nsplit^3 chunks to perform SR, increase nsplit if out of GPU memory, make sure that nsplit devides Ng_lr
model_path="/hildafs/home/xzhangn/state_710.pt" # path of the SR model
style_path="/hildafs/home/xzhangn/xzhangn/sr_pipeline/2-preproc/style/test-set/7_0/set3/PART_099.npy"
lr_input="/hildafs/home/xzhangn/xzhangn/sr_pipeline/2-preproc/preproc_catnorms/test-set/7_0/set3/catnorm_099.npy"
sr_path="/hildafs/home/xzhangn/xzhangn/sr_pipeline/6-makeplots/superres/test-output/PART_099/1"
noise_dir="./noise"

srun python ./lr2sr_custom_noise.py --model-path "$model_path" --style-path $style_path --lr-input "$lr_input" --sr-path "$sr_path" --Lbox-kpc $Boxsize --nsplit $nsplit --noise-dir $noise_dir

date
pwd

