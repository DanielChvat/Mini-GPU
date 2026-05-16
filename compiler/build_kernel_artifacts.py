#!/usr/bin/env python3
"""Build registered Mini-GPU CUDA kernels and write the runtime manifest."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_ROOT = REPO_ROOT / "runtime" / "kernels"
REGISTER_RE = re.compile(
    r"MINIGPU_REGISTER_KERNEL\s*\(\s*"
    r'"([^"]+)"\s*,\s*'
    r'"([^"]+)"\s*,\s*'
    r'"([^"]+)"\s*,\s*'
    r'"([^"]+)"\s*\)'
)
TYPED_REGISTER_RE = re.compile(
    r"MINIGPU_REGISTER_TYPED_KERNELS\s*\(\s*"
    r'"([^"]+)"\s*,\s*'
    r'"([^"]+)"\s*,\s*'
    r'"([^"]+)"\s*\)'
)
DTYPE_TO_COMPILER_DTYPE = {
    "fp8_e4m3fn": "fp8",
}
DTYPE_TO_ENTRY_SUFFIX = {
    "fp8_e4m3fn": "fp8",
}


@dataclass(frozen=True)
class KernelRegistration:
    source: Path
    name: str
    op: str
    dtype: str
    entry: str

    @property
    def artifact_stem(self) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "_", self.name).strip("_")

    @property
    def compiler_dtype(self) -> str:
        return DTYPE_TO_COMPILER_DTYPE.get(self.dtype, self.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile MINIGPU_REGISTER_KERNEL entries and regenerate kernels.yaml."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_KERNEL_ROOT / "src",
        help="Directory containing registered .cu kernels.",
    )
    parser.add_argument(
        "--kernel-root",
        type=Path,
        default=DEFAULT_KERNEL_ROOT,
        help="Runtime kernel artifact root.",
    )
    parser.add_argument("--clang", default="auto", help="Clang++ executable for minigpucc.")
    parser.add_argument("--cuda-path", default="auto", help="CUDA toolkit path for minigpucc.")
    parser.add_argument("--gpu-arch", default="auto", help="CUDA parse arch for minigpucc.")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Only regenerate kernels.yaml from source registrations.",
    )
    return parser.parse_args()


def registered_kernels(source_dir: Path) -> list[KernelRegistration]:
    registrations: list[KernelRegistration] = []
    for source in sorted(source_dir.glob("*.cu")):
        text = source.read_text(encoding="utf-8")
        for match in REGISTER_RE.finditer(text):
            registrations.append(KernelRegistration(source, *match.groups()))
        for match in TYPED_REGISTER_RE.finditer(text):
            op, dtype_text, entry_base = match.groups()
            for dtype in split_dtypes(dtype_text):
                entry_suffix = DTYPE_TO_ENTRY_SUFFIX.get(dtype, dtype)
                registrations.append(
                    KernelRegistration(
                        source=source,
                        name=f"{op}.{dtype}",
                        op=op,
                        dtype=dtype,
                        entry=f"{entry_base}_{entry_suffix}",
                    )
                )

    seen: set[str] = set()
    for registration in registrations:
        if registration.name in seen:
            raise RuntimeError(f"duplicate Mini-GPU kernel registration: {registration.name}")
        seen.add(registration.name)
    return registrations


def split_dtypes(text: str) -> list[str]:
    dtypes = [part.strip() for part in text.split(",") if part.strip()]
    if not dtypes:
        raise RuntimeError("MINIGPU_REGISTER_TYPED_KERNELS requires at least one dtype")
    return dtypes


def compile_registration(
    registration: KernelRegistration,
    *,
    bin_dir: Path,
    map_dir: Path,
    clang: str,
    cuda_path: str,
    gpu_arch: str,
) -> None:
    bin_path = bin_dir / f"{registration.artifact_stem}.bin"
    map_path = map_dir / f"{registration.artifact_stem}.map"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "compiler" / "minigpucc.py"),
        str(registration.source),
        "-o",
        str(bin_path),
        "--save-map",
        str(map_path),
        "--only-kernel",
        registration.entry,
        "--kernel-dtypes",
        registration.compiler_dtype,
        "--clang",
        clang,
        "--cuda-path",
        cuda_path,
        "--gpu-arch",
        gpu_arch,
    ]
    subprocess.run(cmd, check=True)


def manifest(registrations: list[KernelRegistration]) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for registration in registrations:
        stem = registration.artifact_stem
        artifacts.append(
            {
                "file": f"bin/{stem}.bin",
                "map": f"maps/{stem}.map",
                "kernels": [
                    {
                        "name": registration.name,
                        "op": registration.op,
                        "dtype": registration.dtype,
                        "entry": registration.entry,
                    }
                ],
            }
        )

    return {
        "defaults": {
            "grid_dim": 1,
            "block_dim": 4,
            "active_mask": "auto",
            "timeout_ms": 5000,
        },
        "artifacts": artifacts,
    }


def main() -> int:
    args = parse_args()
    kernel_root = args.kernel_root.resolve()
    source_dir = args.source_dir.resolve()
    bin_dir = kernel_root / "bin"
    map_dir = kernel_root / "maps"
    bin_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    registrations = registered_kernels(source_dir)
    if not registrations:
        raise RuntimeError(f"no MINIGPU_REGISTER_KERNEL entries found in {source_dir}")

    if not args.skip_compile:
        for registration in registrations:
            compile_registration(
                registration,
                bin_dir=bin_dir,
                map_dir=map_dir,
                clang=args.clang,
                cuda_path=args.cuda_path,
                gpu_arch=args.gpu_arch,
            )

    manifest_path = kernel_root / "kernels.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest(registrations), sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {manifest_path} with {len(registrations)} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
