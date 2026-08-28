"""Private initial-structure backend + public EADLD physics audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eadld.initialization import DesignSpec, load_backend, run_generation_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="设计指标 -> 私有初始结构 -> EADLD真实追迹")
    parser.add_argument("--efl", type=float, required=True)
    parser.add_argument("--f-number", type=float, required=True)
    parser.add_argument("--half-field", type=float, required=True)
    parser.add_argument("--wavelengths", type=float, nargs="+", required=True)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--min-image-clearance", type=float)
    parser.add_argument("--max-package-length", type=float)
    parser.add_argument("--max-distortion", type=float, help="绝对畸变比例，例如 0.01")
    parser.add_argument("--target-cra", type=float, help="目标主光线角 [deg]")
    parser.add_argument("--min-relative-illumination", type=float)
    parser.add_argument("--max-efl-error", type=float, default=0.05)
    parser.add_argument("--backend", required=True, help="私有 package.module:factory")
    parser.add_argument("--backend-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = DesignSpec(
        args.efl,
        args.f_number,
        args.half_field,
        tuple(args.wavelengths),
        args.elements,
        args.candidate_count,
        args.min_image_clearance,
        args.max_package_length,
        args.max_distortion,
        args.target_cra,
        args.min_relative_illumination,
        args.max_efl_error,
    )
    backend = load_backend(args.backend, args.backend_config)
    manifest = run_generation_audit(spec, backend, args.output_dir)
    selected = next(
        row
        for row in manifest["candidates"]
        if row["candidate_id"] == manifest["selected_candidate_id"]
    )
    print(json.dumps({"status": manifest["status"], "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
