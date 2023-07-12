from math import log2
import numpy as np
import torch
import torch.nn as nn
import math

from .narrow import narrow_by
from .resample import Resampler, Resampler2
from .style import ConvStyled3d, LeakyReLUStyled
from .styled_conv import ResStyledBlock
from .lag2eul import lag2eul



class G(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock(prev_chan, next_chan, out_chan, cat_noise, style_size))

    def forward(self, x, style):
        s = style
        y = x  # direct upsampling from the input

        x = self.block0((x, s))

        # y = None  # no direct upsampling from the input
        for block in self.blocks:
            x, y, s = block(x, y, s)
        print(y.shape, 'output shape')
        return y
    
    
    
class G_styled_noise(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock_styled_noise(prev_chan, next_chan, out_chan, cat_noise, style_size))

    def forward(self, x, style):
        s = style
        y = x  # direct upsampling from the input

        x = self.block0((x, s))

        # y = None  # no direct upsampling from the input
        for block in self.blocks:
            x, y, s = block(x, y, s)
        print(y.shape, 'output shape')
        return y

class G_load_noise(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512, cat_noise=False,
                 **kwargs):
        super().__init__()

        self.style_size = style_size
        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))

        assert chan_min <= chan_max

        def chan(b):
            c = chan_base >> b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan, chan(0), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            prev_chan, next_chan = chan(b), chan(b + 1)
            self.blocks.append(
                HBlock_load_noise(prev_chan, next_chan, out_chan, cat_noise, style_size, layer_id=b))

    def forward(self, x, style,  noise_list=None):
        s = style
        y = x  # direct upsampling from the input

        x = self.block0((x, s))

        # y = None  # no direct upsampling from the input
        for block in self.blocks:
            x, y, s, noise_list = block(x, y, s, noise_list)
        print(y.shape, 'output shape')
        return y


class HBlock(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, out_chan, cat_noise, style_size):
        super().__init__()

        self.upsample = Resampler(3, 2)

        # isolate conv style part to make input right
        self.noise_upsample = nn.Sequential(
            AddNoise(cat_noise, chan=prev_chan),
            self.upsample,
        )

        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise(cat_noise, chan=next_chan) 
        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )

        self.proj = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, y, s):
        x = self.noise_upsample(x)
        x = self.conv((x,s))
        x = self.addnoise(x)
        x = self.conv1((x,s))

        if y is None:
            y = self.proj((x,s))
        else:
            y = self.upsample(y)  

            y = narrow_by(y, 2)
            y = y + self.proj((x,s))
        print(x.shape, y.shape, s.shape, 'hblock output')
        return x, y, s



class HBlock_styled_noise(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, out_chan, cat_noise, style_size):
        super().__init__()

        self.upsample = Resampler(3, 2)

        # isolate conv style part to make input right
        self.noise_upsample = nn.Sequential(
            AddNoise_styled(cat_noise, chan=prev_chan),
            self.upsample,
        )

        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise_styled(cat_noise, chan=next_chan) 
        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )

        self.proj = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, y, s):
        x = self.noise_upsample((x,s))
        x = self.conv((x,s))
        x = self.addnoise((x,s))
        x = self.conv1((x,s))

        if y is None:
            y = self.proj((x,s))
        else:
            y = self.upsample(y)  

            y = narrow_by(y, 2)
            y = y + self.proj((x,s))
        print(x.shape, y.shape, s.shape, 'hblock output')
        return x, y, s

class HBlock_load_noise(nn.Module):
    """The "H" block of the StyleGAN2 generator.

        x_p                     y_p
         |                       |
    convolution           linear upsample
         |                       |
          >--- projection ------>+
         |                       |
         v                       v
        x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958
    Upsampling are all linear, not transposed convolution.

    Parameters
    ----------
    prev_chan : number of channels of x_p
    next_chan : number of channels of x_n
    out_chan : number of channels of y_p and y_n
    cat_noise: concatenate noise if True, otherwise add noise

    Notes
    -----
    next_size = 2 * prev_size - 6
    """

    def __init__(self, prev_chan, next_chan, out_chan, cat_noise, style_size, layer_id=0):
        super().__init__()

        self.upsample = Resampler(3, 2)

        # isolate conv style part to make input right
        self.noise_upsample = nn.Sequential(
            AddNoise_load_noise(cat_noise, prev_chan, layer_id, id_inside=0, use_custom_noise=True),
            self.upsample,
        )

        self.conv = nn.Sequential(
            ConvStyled3d(prev_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )
        self.addnoise = AddNoise_load_noise(cat_noise, next_chan, layer_id, id_inside=1, use_custom_noise=True) 
        self.conv1 = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), next_chan, style_size, 3),
            LeakyReLUStyled(0.2, True),
        )

        self.proj = nn.Sequential(
            ConvStyled3d(next_chan + int(cat_noise), out_chan, style_size, 1),
            LeakyReLUStyled(0.2, True),
        )

    def forward(self, x, y, s, noise_list):
        x = self.noise_upsample((x, noise_list))
        x = self.conv((x,s))
        x = self.addnoise((x, noise_list))
        x = self.conv1((x,s))

        if y is None:
            y = self.proj((x,s))
        else:
            y = self.upsample(y)  

            y = narrow_by(y, 2)
            y = y + self.proj((x,s))
        print(x.shape, y.shape, s.shape, 'hblock output')
        return x, y, s, noise_list


class AddNoise_load_noise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan, layer_id=0, id_inside=0, use_custom_noise=False):
        super().__init__()

        self.cat = cat
        self.layer_id = layer_id
        self.id_inside = id_inside
        self.custom_noise = use_custom_noise
        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, input):
        if self.id_inside == 0:
            x = input[0]
            noise_list = input[1]
        else:
            x = input[0]
            noise_list = input[1]
        if self.custom_noise:
            noise = torch.unsqueeze(noise_list[self.layer_id*2 + self.id_inside], 0)
        else:
            noise = torch.randn_like(x[:, :1])


        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x



class AddNoise(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, chan=1):
        super().__init__()

        self.cat = cat

        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))

    def forward(self, x):
        noise = torch.randn_like(x[:, :1])

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = self.std.view(std_shape) * noise

            x = x + noise

        return x


class AddNoise_styled(nn.Module):
    """Add or concatenate noise.

    Add noise if `cat=False`.
    The number of channels `chan` should be 1 (StyleGAN2)
    or that of the input (StyleGAN).
    """

    def __init__(self, cat, style_size=1, chan=1):
        super().__init__()

        self.cat = cat
        if not self.cat:
            self.std = nn.Parameter(torch.zeros([chan]))
        def init_weight(m):
            if type(m) is nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.ones_(m.bias)
        
        self.style_block = nn.Sequential(
            nn.Linear(in_features=style_size, out_features=32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(in_features=32, out_features=32),
            nn.LeakyReLU(0.2, True),
            nn.Linear(in_features=32, out_features=1),
        )
        self.style_block.apply(init_weight)

    def forward(self, inputs):
        x, s = inputs[0], inputs[1]
        noise = torch.randn_like(x[:, :1])
        
        # change the scale of noise
        s = self.style_block(s)
        std = self.std * s

        if self.cat:
            x = torch.cat([x, noise], dim=1)
        else:
            std_shape = (-1,) + (1,) * (x.dim() - 2)
            noise = std.view(std_shape) * noise

            x = x + noise

        return x



class D(nn.Module):
    def __init__(self, in_chan, out_chan, style_size, scale_factor=16,
                 chan_base=512, chan_min=64, chan_max=512,
                 **kwargs):
        super().__init__()

        self.scale_factor = scale_factor
        num_blocks = round(log2(self.scale_factor))
        self.style_size = style_size

        assert chan_min <= chan_max

        def chan(b):
            if b >= 0:
                c = chan_base >> b
            else:
                c = chan_base << -b
            c = max(c, chan_min)
            c = min(c, chan_max)
            return c

        self.block0 = nn.Sequential(
            ConvStyled3d(in_chan + 1, chan(num_blocks), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        # FIXME here I hard coded the in_chan+1 to meet the dimension after mesh_up factor 1

        self.blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            prev_chan, next_chan = chan(b + 1), chan(b)
            self.blocks.append(ResStyledBlock(in_chan=prev_chan, out_chan=next_chan, style_size=style_size, seq='CACA',
                                              last_act=False))
            self.blocks.append(Resampler2(3, 0.5))

        self.block9 = nn.Sequential(
            ConvStyled3d(chan(0), chan(-1), self.style_size, 1),
            LeakyReLUStyled(0.2, True),
        )
        self.block10 = ConvStyled3d(chan(-1), 1, self.style_size, 1)

    def forward(self, x, style):
        s = style
        # FIXME try do this on GPU
        # rs = torch.clone(s).cpu().numpy()[0][0]
        lag_x = x[:, :3]
        rs = np.float(s)
        eul_x = lag2eul(lag_x, a=rs)[0]
        x = torch.cat([eul_x, x], dim=1)
        x = self.block0((x, s))
        for block in self.blocks:
            x = block((x, s))
        print(x.shape, s.shape, 'shape before block9')
        x = self.block9((x, s))
        x = self.block10((x, s))

        return x
