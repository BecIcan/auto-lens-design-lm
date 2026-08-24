from typing import Callable
import os

import pytorch_lightning as pl
import torch.utils.data
import torchvision
import numpy as np

from eisoptx.utils import utils


class NoneDataModule(pl.LightningDataModule):
    """A dummy datamodule that does not load any data. Used for optimizing lenses in isolation."""

    def __init__(self, n_samples: int = 100):
        """Constructor.

        Args:
            n_samples: Number of samples in a dummy batch when training.
        """
        self.num_workers = 0
        self.n_samples = n_samples
        super().__init__()

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            range(self.n_samples),
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda _: (None,),
        )
        return dataloader

    def test_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            range(1),
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda _: (None,),
        )
        return dataloader


class ImageFolderDataModule(pl.LightningDataModule):
    """A datamodule that loads images from a folder."""

    def __init__(
        self,
        train_folder: str = "data/train_images",
        val_folder: str | None = None,
        tile_layout: tuple[int, int] | None = None,
        batch_size: int = 1,
        val_batch_size: int | None = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
        transforms: list[torch.nn.Module] = (),
        val_transforms: list[torch.nn.Module] | None = None,
    ):
        """Constructor.

        Args:
            train_folder: Path to the folder containing the training images.
            val_folder: Path to the folder containing the validation images; if None, validation is ignored.
            tile_layout: If not None, represents how the field of view is split into tiles (height, width);
                for a given image patch, the corresponding tile is drawn randomly from all possible tiles.
            batch_size: Batch size.
            val_batch_size: Batch size for validation; if None, same as batch_size.
            num_workers: Number of workers for the dataloader.
            persistent_workers: Whether to use persistent workers for the dataloader.
            transforms: List of torchvision transforms to apply to the images.
            val_transforms: List of torchvision transforms to apply to the validation images;
                if None, same as transforms.
        """
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.tile_layout = tile_layout
        self.batch_size = batch_size
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transforms = transforms
        self.dataset_train = None
        self.dataset_val = None
        if self.val_folder is None:
            # If no validation folder, overwrite the method with the default one
            self.val_dataloader = super(pl.LightningDataModule, self).val_dataloader
        if val_transforms is None:
            self.val_transforms = transforms
        else:
            self.val_transforms = val_transforms
        super().__init__()

    def setup(self, stage: str = None):
        transform = torchvision.transforms.Compose(
            [*self.transforms, torchvision.transforms.ToTensor()]
        )
        val_transform = torchvision.transforms.Compose(
            [*self.val_transforms, torchvision.transforms.ToTensor()]
        )
        self.dataset_train = SingleFolderDataset(self.train_folder, transform=transform)
        if self.val_folder is not None:
            self.dataset_val = SingleFolderDataset(
                self.val_folder, transform=val_transform
            )

    def train_dataloader(self):
        collate_fn = FieldLimitCollateFn(self.tile_layout, "pseudorandom")
        dataloader = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def val_dataloader(self):
        # If no validation folder is provided in init, this method is overwritten (workaround)
        # See init
        collate_fn = FieldLimitCollateFn(self.tile_layout, "deterministic")
        dataloader = torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def test_dataloader(self):
        # No image processing in testing mode
        dataloader = torch.utils.data.DataLoader(
            range(1),
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda _: (None,),
            persistent_workers=self.persistent_workers,
        )
        return dataloader


class SingleFolderDataset(torchvision.datasets.VisionDataset):
    """A dataset that loads images from a single folder."""

    def __init__(self, path: str, transform: Callable = None):
        """Constructor.

        Args:
            path: Path to the folder containing the images.
            transform: Torchvision transform to apply to the images.
        """
        super().__init__(path, transform=transform, target_transform=None)
        self.samples = []
        for root, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)
        self.loader = torchvision.datasets.folder.default_loader

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, index

    def __len__(self):
        return len(self.samples)


class FieldLimitCollateFn:
    """Collate function for the dataloader that returns the images and the field limits."""

    def __init__(self, tile_layout: tuple[int, int] | None, tile_sampling_mode: str):
        """Constructor.

        Args:
            tile_layout: If not None, represents how the field of view is split into tiles (height, width);
                for a given image patch, the corresponding tile is drawn randomly from all possible tiles.
            tile_sampling_mode: Strategy for sampling tiles; 'random', 'pseudorandom' or 'deterministic'.
        """
        self.tile_layout = tile_layout
        self.tile_sampling_mode = tile_sampling_mode
        # List all tiles and estimate their radial distance in relative coordinates
        tile_idx = np.stack(
            (
                np.broadcast_arrays(
                    np.arange(tile_layout[0])[:, None],
                    np.arange(tile_layout[1])[None, :],
                )
            ),
            axis=-1,
        )
        tile_layout = np.array(tile_layout)
        # Sort tiles by radial distance in reverse order
        tile_radial_distance = np.linalg.norm(tile_idx + 0.5 - tile_layout / 2, axis=-1)
        tile_order = np.argsort(-tile_radial_distance.flatten())
        self.tile_order = torch.tensor(tile_order)

    def __call__(self, batch):
        n_samples = len(batch)
        images, indices = zip(*batch)
        images = torch.stack(images, dim=0)
        indices = torch.tensor(indices)
        if self.tile_layout is not None:
            # Sample from tile layout
            n_rows, n_cols = self.tile_layout
            n_tiles = n_rows * n_cols
            if self.tile_sampling_mode == "random":
                soft_relative_indices = torch.rand(n_samples)
            elif self.tile_sampling_mode == "pseudorandom":
                # Sample tiles pseudo-randomly as to sample different field regions for each sample in a batch
                soft_relative_indices = (
                    torch.arange(n_samples) + torch.rand((n_samples,))
                ) / n_samples
            elif self.tile_sampling_mode == "deterministic":
                partial_index = indices % (n_samples + 1) / n_samples  # 0 to 1
                soft_relative_indices = (
                    torch.arange(n_samples) + partial_index
                ) / n_samples
            else:
                raise ValueError(
                    f"Invalid tile_sampling_mode: {self.tile_sampling_mode}"
                )
            field_indices = (
                (soft_relative_indices * n_tiles).to(int).clamp(max=n_tiles - 1)
            )
            i = self.tile_order[field_indices]
            field_lims = utils.compute_field_lims(i, n_rows, n_cols)
            return [images, field_lims]
        else:
            return [images, None]
