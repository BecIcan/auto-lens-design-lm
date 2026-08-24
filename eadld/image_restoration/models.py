import torch


class ImageRestorationChainer(torch.nn.Module):
    """Image restoration model composed of an arbitrary sequence of image restoration models."""

    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, im, psf_grid):
        """Apply the image restoration model to a blurred image.

        From the PSF grid, compute an effective, approximate PSF for the whole image.

        Args:
            im: Blurred image (shape: [B, C, H, W]).
            psf_grid: Point spread function (shape: [B, P, C, H, W]).
        """
        # Compute the effective PSF by averaging the PSF grid
        if psf_grid.dim() == 5:
            effective_psf = psf_grid.mean(dim=1)
        else:
            effective_psf = psf_grid
        assert effective_psf.dim() == 4, "PSF grid must have 4 dimensions"

        # Apply the image restoration models in sequence
        for model in self.models:
            im = model(im, effective_psf)

        return im
