import numpy as np
import torch
from bigfile import BigFile
from map2map.norms import cosmology


def dis2pos(dis_field, boxsize, Ng):
    """Assume 'dis_field' is in order of `pid` that aligns with the Lagrangian lattice,
    and dis_field.shape = (3,Ng,Ng,Ng)
    """
    cellsize = boxsize / Ng
    lattice = np.arange(Ng) * cellsize + 0.5 * cellsize

    pos = dis_field.copy()

    pos[2] += lattice
    pos[1] += lattice.reshape(-1, 1)
    pos[0] += lattice.reshape(-1, 1, 1)

    pos[pos < 0] += boxsize
    pos[pos > boxsize] -= boxsize

    return pos


def narrow_like(sr_box, tgt_Ng):
    """ sr_box in shape (Nc,Ng,Ng,Ng),trim to (Nc,tgt_Ng,tgt_Ng,tgt_Ng), better to be even """
    width = np.shape(sr_box)[1] - tgt_Ng
    half_width = width // 2
    begin, stop = half_width, tgt_Ng + half_width
    return sr_box[:, begin:stop, begin:stop, begin:stop]


def cropfield(field, idx, reps, crop, pad):
    """input field in shape of (Nc,Ng,Ng,Ng),
    crop idx^th subbox in reps grid with padding"""
    start = np.unravel_index(idx, reps) * crop  # find coordinate of idx in reps grid
    x = field.copy()
    for d, (i, N, (p0, p1)) in enumerate(zip(start, crop, pad)):
        x = x.take(range(i - p0, i + N + p1), axis=1 + d, mode='wrap')
    
    return x


def sr_field_custom_noise(model, lr_field, style, tgt_size, noise_list):
    """input *normalized* lr_field in shape of (Nc,Ng,Ng,Ng),
    return unnormalized sr_field trimmed to tgt_size^3
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_field = np.expand_dims(lr_field, axis=0)
    lr_field = torch.from_numpy(lr_field).float().to(device)
    style = torch.from_numpy(style).float().to(device)
    model.to(device)
    for i, x in enumerate(noise_list):
        x = torch.from_numpy(x).float().to(device)
        noise_list[i] = x.to(device)

    with torch.no_grad():
        sr_box = model(lr_field, style, noise_list)

    sr_box = sr_box.cpu().numpy()
    sr_disp = cosmology.disnorm(sr_box[0, 0:3, ], a=style.cpu().numpy(), undo=True)
    sr_disp = narrow_like(sr_disp, tgt_size)
    sr_vel = cosmology.velnorm(sr_box[0, 3:6, ], a=style.cpu().numpy(), undo=True)
    sr_vel = narrow_like(sr_vel, tgt_size)
    return sr_disp, sr_vel


def sr_field(model, lr_field, style, tgt_size):
    """input *normalized* lr_field in shape of (Nc,Ng,Ng,Ng),
    return unnormalized sr_field trimmed to tgt_size^3
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_field = np.expand_dims(lr_field, axis=0)
    lr_field = torch.from_numpy(lr_field).float().to(device)
    style = torch.from_numpy(style).float().to(device)
    model.to(device)

    with torch.no_grad():
        sr_box = model(lr_field, style)

    sr_box = sr_box.cpu().numpy()
    sr_disp = cosmology.disnorm(sr_box[0, 0:3, ], a=style.cpu().numpy(), undo=True)
    sr_disp = narrow_like(sr_disp, tgt_size)
    sr_vel = cosmology.velnorm(sr_box[0, 3:6, ], a=style.cpu().numpy(), undo=True)
    sr_vel = narrow_like(sr_vel, tgt_size)
    return sr_disp, sr_vel


def crop_noise(size, idx, pad, n_split=4, noise_dir=None):
    """
    crop info: idx, reps, crop, pad 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_path = noise_dir + '/noise_{}.npy'.format(size)
    noise = np.load(noise_path)
    _, reps, crop, pad = crop_info(noise, n_split, pad)
    chunk = cropfield(noise, idx, reps, crop, pad)
    return chunk

def gen_ids(Ng):
    return np.arange(Ng**3)


def crop_info(noise, n_split, pad):
    """return the crop info of the noise field"""
    size = noise.shape[1:]
    size = np.asarray(size)
    chunk_size = size // n_split
    crop = np.broadcast_to(chunk_size, size.shape)
    reps = size // crop
    tot_reps = int(np.prod(reps))
    ndim = len(size)
    pad = np.broadcast_to(pad, (ndim,2))
    return tot_reps, reps, crop, pad 