import warnings
import sys

from pytorch_lightning import cli, loggers, utilities, callbacks
import torch
import numpy as np

from eadld.imaging_system import ImagingSystemModule


class CustomLogger(loggers.TensorBoardLogger):
    """Custom logger that removes epoch from metrics."""

    @utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        # Remove epoch logger
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)


class CustomCLI(cli.LightningCLI):
    """Custom CLI that links arguments for the ImagingSystemModule."""

    def add_arguments_to_parser(self, parser):
        # Name of experiment corresponds to the lens sequence
        parser.link_arguments(
            "model.lens_parameterization.init_args.lens_sequence",
            "trainer.logger.init_args.name",
        )
        # Sensor diagonal from EFL and HFOV
        parser.link_arguments(
            (
                "model.lens_parameterization.init_args.target_efl",
                "model.ray_initialization.init_args.hfov",
            ),
            "model.optics_simulator.init_args.sensor_diagonal",
            lambda efl, hfov: float(2 * efl * np.tan(np.deg2rad(hfov))),
        )
        # Wavelengths for the optics simulator
        parser.link_arguments(
            "model.ray_initialization.init_args.wavelengths",
            "model.optics_simulator.init_args.wavelengths",
        )

    def fit(self, model, **kwargs):
        # TODO: Try again once Windows is supported
        # model = torch.compile(model)
        self.trainer.fit(model, **kwargs)


class CustomProgressBar(callbacks.TQDMProgressBar):
    """Custom progress bar that disables itself when not running in a terminal."""

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    torch.set_float32_matmul_precision("high")
    lightning_cli = CustomCLI(ImagingSystemModule, auto_configure_optimizers=False)
