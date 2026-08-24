import numpy as np
from pytorch_lightning import loggers
import pytorch_lightning as pl
import torch


def apply_recursively(obj: any, fn: callable):
    """Apply a function recursively to a nested dictionary, list or tuple.

    Args:
        obj: The nested dictionary, list or tuple, or a scalar.
        fn: The function to apply.
    """
    if isinstance(obj, tuple):
        return (apply_recursively(item, fn) for item in obj)
    elif isinstance(obj, list):
        return [apply_recursively(item, fn) for item in obj]
    elif isinstance(obj, dict):
        return {k: apply_recursively(v, fn) for k, v in obj.items()}
    else:
        return fn(obj)


def retrieve_log_dir(trainer: pl.Trainer):
    """Retrieve the TensorBoard log directory from the trainer.

    Args:
        trainer: The PyTorch Lightning trainer.
    """
    log_dir = None
    for logger in trainer.loggers:
        if isinstance(logger, loggers.TensorBoardLogger):
            tb_logger = logger.experiment
            log_dir = tb_logger.log_dir
            break
    return log_dir


def compute_field_lims(tile_idx: int | torch.Tensor, n_rows: int, n_cols: int):
    """Assuming the optical is split into a grid of tiles, return the field limits of a tile in relative coordinates.

    Args:
        tile_idx: Index of the tile.
        n_rows: Number of rows in the tile grid.
        n_cols: Number of columns in the tile grid.
    """
    if isinstance(tile_idx, int):
        tile_idx = torch.tensor([tile_idx])
    assert all(tile_idx < n_rows * n_cols), (
        f"Tile index {tile_idx} is out of bounds for a {n_rows}x{n_cols} grid."
    )

    row = tile_idx // n_cols
    col = tile_idx % n_cols

    diag = np.sqrt(n_rows**2 + n_cols**2)
    x_lims = n_cols / diag * (2 * torch.stack((col, col + 1), dim=-1) / n_cols - 1)
    y_lims = -n_rows / diag * (2 * torch.stack((row, row + 1), dim=-1) / n_rows - 1)
    field_lims = torch.cat((x_lims, y_lims), dim=-1)
    return field_lims
