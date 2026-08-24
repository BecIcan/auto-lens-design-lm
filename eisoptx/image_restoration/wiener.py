import math

import torch
import torch.nn.utils.parametrize


class WienerDeconvolution(torch.nn.Module):
    """Wiener deconvolution model."""

    def __init__(
        self,
        learn_snr: bool = True,
        snr_initial_value: float = 1.0,
        detach_psfs: bool = False,
    ):
        """Constructor.

        Args:
            learn_snr: Whether to learn the signal-to-noise ratio.
            snr_initial_value: Initial value of the signal-to-noise ratio.
            detach_psfs: Whether to detach the PSFs from the computational graph.
        """
        super().__init__()
        if learn_snr:
            self.snr = torch.nn.Parameter(torch.tensor(math.log(snr_initial_value)))
            torch.nn.utils.parametrize.register_parametrization(self, "snr", Exp())
        else:
            self.snr = None
        self.detach_psfs = detach_psfs

    def forward(self, im: torch.Tensor, psf_grid: torch.Tensor):
        """Apply Wiener deconvolution to a blurred image.

        From the PSF grid, we compute an effective, approximate PSF for the whole image.

        Args:
            im: Blurred image (shape: [B, C, H, W]).
            psf_grid: Point spread function (shape: [B, P, C, H, W]).
        """
        if self.detach_psfs:
            psf_grid = psf_grid.detach()
        # Compute the effective PSF by averaging the PSF grid
        if psf_grid.dim() == 5:
            effective_psf = psf_grid.mean(dim=1)
        else:
            effective_psf = psf_grid
        assert effective_psf.dim() == 4, "PSF grid must have 4 dimensions"

        # Compute a reasonable approximation for SNR if not provided
        if self.snr is None:
            median_im = median_filter(im, kernel_size=3)
            noise_im = im - median_im
            noise_var = noise_im.var()
            signal_var = median_im.var()
            snr = (signal_var / noise_var).item()
        else:
            snr = self.snr

        im_out = wiener_deconvolution(im, effective_psf, snr)

        return im_out


class Exp(torch.nn.Module):
    """Exponential transformation used for parametrization."""

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return x.exp()


def wiener_deconvolution(im: torch.Tensor, kernel: torch.Tensor, snr: float):
    """Perform Wiener deconvolution on a blurred image.

    Args:
        im: Blurred image (shape: [..., C, H, W]).
        kernel: Blurring kernel (shape: [..., C, H, W]).
        snr: Signal-to-noise ratio.
    """
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    im_h, im_w = im.shape[-2:]

    # Pad the image and kernel
    padded_im = torch.nn.functional.pad(
        im, (pad_w, pad_w, pad_h, pad_h), mode="replicate"
    )
    padded_kernel = torch.nn.functional.pad(
        kernel,
        (
            0,
            padded_im.shape[-1] - kernel.shape[-1],
            0,
            padded_im.shape[-2] - kernel.shape[-2],
        ),
        mode="constant",
    )
    padded_kernel = padded_kernel.roll(shifts=(-pad_h, -pad_w), dims=(-2, -1))

    # Fourier transform of the blurred image and the padded PSF
    im_f = torch.fft.rfft2(padded_im)
    kernel_f = torch.fft.rfft2(padded_kernel)

    # Compute the Wiener filter in the frequency domain
    wiener_f = kernel_f.conj() / (kernel_f.abs() ** 2 + 1.0 / snr)

    # Deblur the image in the frequency domain
    im_out_f = im_f * (wiener_f - 1)

    # Inverse Fourier transform to get the deblurred image
    im_out = torch.fft.irfft2(im_out_f)
    im_out = im_out[..., pad_h : pad_h + im_h, pad_w : pad_w + im_w]

    return im_out + im


def binomial_filter(image, kernel_size: int = 3):
    """Apply a binomial filter to an image (approximation of Gaussian).

    Uses two separable convolutions.

    Args:
        image: Image (shape: [..., H, W]).
        kernel_size: Size of the binomial filter.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # Pad the image
    pad = kernel_size // 2
    padded = torch.nn.functional.pad(image, (pad,) * 4, mode="reflect")

    # Compute binomial filter
    coefficients = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
    binomial_filter = torch.tensor(coefficients).to(image) / (kernel_size - 1) ** 2

    # Apply binomial filter in two separable convolutions
    conved1 = torch.nn.functional.conv2d(
        padded.view(-1, 1, *padded.shape[-2:]), binomial_filter.view(1, 1, 1, -1)
    )
    conved2 = torch.nn.functional.conv2d(conved1, binomial_filter.view(1, 1, -1, 1))

    return conved2.view(image.shape)


def median_filter(image, kernel_size: int = 3):
    """Apply a median filter to an image.

    Args:
        image: Image (shape: [..., H, W]).
        kernel_size: Size of the median filter.
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # Pad the image
    pad = kernel_size // 2
    padded = torch.nn.functional.pad(image, (pad,) * 4, mode="reflect")

    # Extract patches
    patches = (
        padded.unfold(2, kernel_size, 1)
        .unfold(3, kernel_size, 1)
        .reshape(*image.shape, -1)
    )

    # Compute median
    median_vals, _ = patches.median(dim=-1)

    return median_vals
