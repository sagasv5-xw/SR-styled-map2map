import itertools
import torch

from ..norms.cosmology import D




def pixel_shuffle_3d_inv(x, r):
    """
    Rearranges tensor x with shape ``[B,C,H,W,D]`` 
    to a tensor of shape ``[B,C*r*r*r,H/r,W/r,D/r]``.
    """
    [B, C, H, W, D] = list(x.size())
    x = x.contiguous().view(B, C, H//r, r, W//r, r, D//r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.contiguous().view(B, C*(r**3), H//r, W//r, D//r)
    return x


def lag2eul(
        dis,
        val=1.0,
        eul_scale_factor=2,
        eul_pad=0,
        rm_dis_mean=True,
        periodic=False,
        a=0.0,
        dis_std=6.0,
        boxsize=100.,
        meshsize=512,
        **kwargs):
    """Transform fields from Lagrangian description to Eulerian description

    Only works for 3d fields, output same mesh size as input.

    Use displacement fields `dis` to map the value fields `val` from Lagrangian
    to Eulerian positions and then "paint" with CIC (trilinear) scheme.
    Displacement and value fields are paired when are sequences of the same
    length. If the displacement (value) field has only one entry, it is shared
    by all the value (displacement) fields.

    The Eulerian size is scaled by the `eul_scale_factor` and then padded by
    the `eul_pad`.

    Common mean displacement of all inputs can be removed to bring more
    particles inside the box. Periodic boundary condition can be turned on.

    Note that the box and mesh sizes don't have to be that of the inputs, as
    long as their ratio gives the right resolution. One can therefore set them
    to the values of the whole Lagrangian fields, and use smaller inputs.

    Implementation follows pmesh/cic.py by Yu Feng.
    """
    # NOTE the following factor assumes the displacements have been normalized
    # by data.norms.cosmology.dis, and thus undoes it
    z = 1/a - 1

    dis_norm = dis_std * D(z) * meshsize / boxsize  # to mesh unit
    dis_norm *= eul_scale_factor

    if isinstance(dis, torch.Tensor):
        dis = [dis]
    if isinstance(val, (float, torch.Tensor)):
        val = [val]
    if len(dis) != len(val) and len(dis) != 1 and len(val) != 1:
        raise ValueError('dis-val field mismatch')

    if any(d.dim() != 5 for d in dis):
        raise NotImplementedError('only support 3d fields for now')
    if any(d.shape[1] != 3 for d in dis):
        raise ValueError('only support 3d displacement fields')

    # common mean displacement of all inputs
    # if removed, fewer particles go outside of the box
    # common for all inputs so outputs are comparable in the same coords
    d_mean = 0
    if rm_dis_mean:
        d_mean = sum(d.detach().mean((2, 3, 4), keepdim=True)
                     for d in dis) / len(dis)

    out = []
    if len(dis) == 1 and len(val) != 1:
        dis = itertools.repeat(dis[0])
    elif len(dis) != 1 and len(val) == 1:
        val = itertools.repeat(val[0])
    for d, v in zip(dis, val):
        dtype, device = d.dtype, d.device

        N, DHW = d.shape[0], d.shape[2:]
        DHW = torch.Size([s * eul_scale_factor + 2 * eul_pad for s in DHW])

        if isinstance(v, float):
            C = 1
        else:
            C = v.shape[1]
            v = v.contiguous().flatten(start_dim=2).unsqueeze(-1)

        mesh = torch.zeros(N, C, *DHW, dtype=dtype, device=device)

        pos = (d - d_mean) * dis_norm
        del d

        pos[:, 0] += torch.arange(0.5, DHW[0] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)[:, None, None]
        pos[:, 1] += torch.arange(0.5, DHW[1] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)[:, None]
        pos[:, 2] += torch.arange(0.5, DHW[2] - 2 * eul_pad, eul_scale_factor,
                                  dtype=dtype, device=device)

        pos = pos.contiguous().view(N, 3, -1, 1)  # last axis for neighbors

        intpos = pos.floor().to(torch.int)
        neighbors = (
            torch.arange(8, device=device)
            >> torch.arange(3, device=device)[:, None, None]
        ) & 1
        tgtpos = intpos + neighbors
        del intpos, neighbors

        # CIC
        kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)
        del pos

        v = v * kernel
        del kernel

        tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
        v = v.view(N, C, -1)

        for n in range(N):  # because ind has variable length
            bounds = torch.tensor(DHW, device=device)[:, None]

            if periodic:
                torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

            ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]
                   ) * DHW[2] + tgtpos[n, 2]
            src = v[n]

            if not periodic:
                mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                ind = ind[mask]
                src = src[:, mask]

            mesh[n].view(C, -1).index_add_(1, ind, src)
        
        if eul_scale_factor > 1:
            #print(mesh.shape,'before shuffle')
            mesh = pixel_shuffle_3d_inv(mesh, eul_scale_factor)
            #print(mesh.shape,'after shuffle')

        out.append(mesh)

    return out


def lag2eul_old(*xs, up_fac = 1, squeeze=False, rm_dis_mean=True, periodic=False):
    """
    Transform fields from Lagrangian description to Eulerian description

    Only works for 3d fields, output same mesh size as input.

    Input of shape `(N, C, ...)` is first split into `(N, 3, ...)` and
    `(N, C-3, ...)`. Take the former as the displacement field to map the
    latter from Lagrangian to Eulerian positions and then "paint" with CIC
    (trilinear) scheme. Use 1 if the latter is empty.

    Implementation follows pmesh/cic.py by Yu Feng.
    
    up_fac  : up_fac > 1, allows to paint to a higher resolution mesh
    squeeze : If True, apply inverse pixel shuffle to match the DHW size 
    
    """
    # FIXME for other redshift, box and mesh sizes
    z = 2
    Boxsize = 100
    Nmesh = 512 * up_fac  # the mesh to paint the particles
    dis_norm = 6 * D(z) * Nmesh / Boxsize  # to mesh unit

    if any(x.dim() != 5 for x in xs):
        raise NotImplementedError('only support 3d fields for now')
    if any(x.shape[1] < 3 for x in xs):
        raise ValueError('displacement not available with <3 channels')

    # common mean displacement of all inputs
    # if removed, fewer particles go outside of the box
    # common for all inputs so outputs are comparable in the same coords
    dis_mean = 0
    if rm_dis_mean:
        dis_mean = sum(x[:, :3].detach().mean((2, 3, 4), keepdim=True)
                       for x in xs) / len(xs)

    out = []
    for x in xs:
        N, Cin, DHW = x.shape[0], x.shape[1], x.shape[2:]
        
        if up_fac > 1:
            DHW = torch.Size([x*up_fac for x in DHW])
            
        if Cin == 3:
            Cout = 1
            val = 1
        else:
            Cout = Cin - 3
            val = x[:, 3:].contiguous().view(N, Cout, -1, 1)
        mesh = torch.zeros(N, Cout, *DHW, dtype=x.dtype, device=x.device)

        pos = (x[:, :3] - dis_mean) * dis_norm

        pos[:, 0] += torch.arange(0.5, DHW[0], up_fac, device=x.device)[:, None, None]
        pos[:, 1] += torch.arange(0.5, DHW[1], up_fac, device=x.device)[:, None]
        pos[:, 2] += torch.arange(0.5, DHW[2], up_fac, device=x.device)

        pos = pos.contiguous().view(N, 3, -1, 1)

        intpos = pos.floor().to(torch.int)
        neighbors = (torch.arange(8, device=x.device)
            >> torch.arange(3, device=x.device)[:, None, None] ) & 1
        tgtpos = intpos + neighbors
        del intpos, neighbors

        # CIC
        kernel = (1.0 - torch.abs(pos - tgtpos)).prod(1, keepdim=True)
        del pos

        val = val * kernel
        del kernel

        tgtpos = tgtpos.view(N, 3, -1)  # fuse spatial and neighbor axes
        val = val.view(N, Cout, -1)

        for n in range(N):  # because ind has variable length
            bounds = torch.tensor(DHW, device=x.device)[:, None]

            if periodic:
                torch.remainder(tgtpos[n], bounds, out=tgtpos[n])

            ind = (tgtpos[n, 0] * DHW[1] + tgtpos[n, 1]
                   ) * DHW[2] + tgtpos[n, 2]
            src = val[n]

            if not periodic:
                mask = ((tgtpos[n] >= 0) & (tgtpos[n] < bounds)).all(0)
                ind = ind[mask]
                src = src[:, mask]

            mesh[n].view(Cout, -1).index_add_(1, ind, src)
            
        
        if (up_fac>1 & squeeze == True):
            mesh = pixel_shuffle_3d_inv(mesh, up_fac)
            
        out.append(mesh)

    return out