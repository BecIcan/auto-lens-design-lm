"""运行唯一公开的三片环带镜头自动优化示例。"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # 三个配置依次定义系统、初始处方和受约束 LM；命令行只负责预算。
    subprocess.run(
        [
            sys.executable,
            "-m",
            "eadld.main",
            "fit",
            "-c",
            "configs/multi_element/defaults.yml",
            "-c",
            "configs/annular_triplet/designs/cooke_annular_initial.yml",
            "-c",
            "configs/multi_element/stage_zone_3p_fold_cooke_f28_annular_m90.yml",
            "--trainer.max_steps=800",
            "--data.init_args.n_samples=800",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
