import torch
import torch.nn as nn


class NAFNet(nn.Module):
    """NAFNet model for image restoration."""

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=(),
        dec_blk_nums=(),
    ):
        """Constructor.

        Args:
            img_channel: Number of input image channels.
            width: Width of the network.
            middle_blk_num: Number of blocks in the middle of the network.
            enc_blk_nums: Number of blocks in each encoder level.
            dec_blk_nums: Number of blocks in each decoder level.
        """
        super().__init__()

        conv_kwargs = dict(kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.intro = nn.Conv2d(
            in_channels=img_channel, out_channels=width, **conv_kwargs
        )
        self.ending = nn.Conv2d(
            in_channels=width, out_channels=img_channel, **conv_kwargs
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor, *args):
        """Forward pass.

        Args:
            inp: Input image(s) (shape: [B, C, H, W]).
        """
        b, c, h, w = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :h, :w]

    def check_image_size(self, x: torch.Tensor):
        """Check if the image size is divisible by 2 multiple times."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFBlock(nn.Module):
    """NAFNet block."""

    def __init__(
        self,
        c: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        """Constructor.

        Args:
            c: Number of input channels.
            dw_expand: Depthwise expansion factor.
            ffn_expand: Feedforward network expansion factor.
            drop_out_rate: Dropout rate.
        """
        super().__init__()
        dw_channel = c * dw_expand
        conv1x1_kwargs = dict(kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, **conv1x1_kwargs)
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, **conv1x1_kwargs
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                **conv1x1_kwargs,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, **conv1x1_kwargs
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, **conv1x1_kwargs
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor):
        """Forward pass.

        Args:
            inp: Input tensor (shape: [B, C, H, W]).
        """
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class SimpleGate(nn.Module):
    """SimpleGate module."""

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor (shape: [B, C, H, W]).
        """
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """Layer normalization module."""

    def __init__(self, channels: int, eps: float = 1e-6):
        """Constructor.

        Args:
            channels: Number of input channels.
            eps: Epsilon value for numerical stability.
        """
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        n, c, h, w = x.size()

        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / torch.sqrt(var + self.eps)

        y = self.weight.view(1, c, 1, 1) * y + self.bias.view(1, c, 1, 1)

        return y
