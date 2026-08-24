from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="lm-annular",
    version="0.1.0",
    packages=find_packages(include=("eisoptx", "eisoptx.*")),
    license="MIT",
    description="Differentiable annular optical design with LM and RayWave validation",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    install_requires=[
        "jsonargparse[signatures]>4.20.0",
        "lightning",
        "matplotlib",
        "numpy",
        "pandas",
        "pillow",
        "pyyaml",
        "scipy",
        "tensorboard",
        "torch",
        "torchmetrics",
        "torchvision",
    ],
    extras_require={"dev": ["pytest"]},
)
