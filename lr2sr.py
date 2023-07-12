import numpy as np
import torch
from map2map.models import styled_srsgan
from utils import *
from bigfile import BigFile

import argparse
import os

# Load parameter and model
# --------------------------
parser = argparse.ArgumentParser(description='lr2sr')
parser.add_argument('--model-path', required=True, type=str, help='path of the generative model')
parser.add_argument('--style-path', required=True, type=str, help='path to the style of input, in this case its a numpy array')
parser.add_argument('--lr-input', required=True, type=str, help='path of the lr input')
parser.add_argument('--sr-path', required=True, type=str, help='path to save sr output')
parser.add_argument('--Lbox-kpc', default=100000, type=float, help='LR/HR/SR Boxsize, in kpc/h')
parser.add_argument('--nsplit', default=4, type=int, help='split the LR box into chunks to apply SR')



args = parser.parse_args()

# some parameters
upsample_fac = 8
in_channels = out_channels = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_split = args.nsplit
# --------------------------
# paths
sr_path = args.sr_path
model_path = args.model_path
# --------------------------
lr_box = np.load(args.lr_input) # lr_input in shape of (Nc,Ng,Ng,Ng)
style = np.load(args.style_path) # style in shape of (1,1)
style_size = style.shape
# --------------------------
tot_reps, reps, crop, pad = crop_info(lr_box, n_split, 3)

size = lr_box.shape[1:]
size = np.asarray(size)
chunk_size = size // n_split

tgt_size = crop[0] * upsample_fac
tgt_chunk = np.broadcast_to(tgt_size, size.shape)

Ng_sr = size[0] * upsample_fac
disp_field = np.zeros([3, Ng_sr, Ng_sr, Ng_sr])
vel_field = np.zeros([3, Ng_sr, Ng_sr, Ng_sr])

# load model
model = styled_srsgan.G(in_channels, out_channels, style_size, upsample_fac)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state['model'])
print('load model state at epoch {}'.format(state['epoch']))
epoch = state['epoch']
del state
model.eval()

for idx in range(0, tot_reps):
    lr_chunk = cropfield(lr_box, idx, reps, crop, pad)
    chunk_disp, chunk_vel = sr_field(model, lr_chunk, style, tgt_size)
    ns = np.unravel_index(idx, reps) * tgt_chunk  # new start point
    disp_field[:, ns[0]:ns[0] + tgt_size, ns[1]:ns[1] + tgt_size, ns[2]:ns[2] + tgt_size] = chunk_disp
    vel_field[:, ns[0]:ns[0] + tgt_size, ns[1]:ns[1] + tgt_size, ns[2]:ns[2] + tgt_size] = chunk_vel
    print("{}/{} done".format(idx + 1, tot_reps), flush=True)

disp_field = np.float32(disp_field)
vel_field = np.float32(vel_field)

# save sr output
Lbox = args.Lbox_kpc
pos_field = dis2pos(disp_field, Lbox, Ng_sr)

sr_pos = pos_field.reshape(3, Ng_sr * Ng_sr * Ng_sr).transpose()
sr_vel = vel_field.reshape(3, Ng_sr * Ng_sr * Ng_sr).transpose()


path = args.sr_path
os.makedirs(path, exist_ok=True)

dest = BigFile(path, create=1)

blockname = 'Position'
dest.create_from_array(blockname, sr_pos)


blockname = 'Velocity'
dest.create_from_array(blockname, sr_vel)

print("Generated SR column in ", path)
